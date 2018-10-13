"""Training manager and multi optimizer for training PyTorch models"""
import json
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchnet as tnt
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable


from networks import VGG16_bn, Classifier, VGG16_bn_finetuned
from dataset import loader_bu3dfe, loader_tensor


class RankLoss(nn.Module):
    def __init__(self, margin):
        super(RankLoss, self).__init__()
        self.margin = margin

    def forward(self, scores1, scores2, target):
        ps1 = F.softmax(scores1)[:, target.long().data]
        ps2 = F.softmax(scores2)[:, target.long().data]
        return torch.clamp(ps1 - ps2 + self.margin, min=0)


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model):
        """
        Initializes training manager with training args and model
        :param args: argparse training args from command line
        :param model: PyTorch model to train
        """
        self.args = args

        self.cuda = args.cuda
        self.model = model

        # Set up data loader, criterion, and pruner.
        train_loader = dataset.train_loader_cubs
        test_loader = dataset.test_loader_cubs
        self.train_data_loader = train_loader(args.train_path,
            args.batch_size, pin_memory=args.cuda, flipcrop=True)
        self.test_data_loader = test_loader(args.test_path,
            args.batch_size, pin_memory=args.cuda, flipcrop=True)
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_rank = RankLoss(0.05)

    def eval(self):
        """Performs evaluation."""
        self.model.eval()
        error_meters = None

        print('Performing eval...')
        for batch, label in tqdm(self.test_data_loader, desc='Eval'):
            if self.cuda:
                batch = batch.cuda()
            batch = Variable(batch, volatile=True)
            scores = self.model(batch)
            # Init error meter.
            outputs = [score.data.view(-1, score.size(1)) for score in scores]
            label = label.view(-1)
            if error_meters is None:
                topk = [1]
                if outputs[0].size(1) > 5:
                    topk.append(5)
                error_meters = [tnt.meter.ClassErrorMeter(
                    topk=topk) for _ in outputs]
            for error_meter, output in zip(error_meters, outputs):
                error_meter.add(output, label)

        errors = [error_meter.value() for error_meter in error_meters]
        for i, error in enumerate(errors):
            print('Scale {} Error: '.format(i+1) +
                  ', '.join('@%s=%.2f' % t for t in zip(topk, error)))
        self.model.train()

        return errors

    def do_batch(self, optimizer, batch, label, optimize_class=True):
        """
        Runs model for one batch
        :param optimizer: Optimizer for training
        :param batch: (num_batch, 3, h, w) Torch tensor of data
        :param label: (num_batch) Torch tensor of classes
        """
        if self.cuda:
            batch = batch.cuda()
            label = label.cuda()
        batch = Variable(batch)
        label = Variable(label)

        # Set grads to 0.
        self.model.zero_grad()
        # Do forward-backward.
        scores = self.model(batch)
        if optimize_class:
            for i in range(len(scores)-1, -1, -1):
                if optimize_class:
                    retain_graph = i > 0
                    self.criterion_class(scores[i], label).backward(
                        retain_graph=retain_graph)
        else:
            for i in range(len(scores)-1, 0, -1):
                retain_graph = (i-1) > 0
                self.criterion_rank(
                    scores[i-1], scores[i], label).backward(retain_graph=retain_graph)

        # Update params.
        optimizer.step()

    def do_epoch(self, epoch_idx, optimizer, optimize_class=True):
        """
        Trains model for one epoch
        :param epoch_idx: int epoch number
        :param optimizer: Optimizer for training
        """
        for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
            self.do_batch(optimizer, batch, label,
                          optimize_class=optimize_class)

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        # Prepare the ckpt.
        self.model.cpu()
        ckpt = {
            'args': self.args,
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'state_dict': self.model.state_dict(),
        }
        if self.cuda:
            self.model.cuda()

        # Save to file.
        torch.save(ckpt, savename + '.pt')

    def load_model(self, savename):
        """
        Loads model from a saved model pt file
        :param savename: string file prefix
        """
        ckpt = torch.load(savename + '.pt')
        self.model.load_state_dict(ckpt['state_dict'])
        self.args = ckpt['args']

    def train(self, epochs, cnn_optimizer, apn_optimizer, savename='', best_accuracy=0):
        """Performs training."""
        best_accuracy = best_accuracy
        error_history = []
        optimize_class = True
        class_epoch = 0
        rank_epoch = 0
        # self.model.flip_apn_grads()

        if self.args.cuda:
            self.model = self.model.cuda()

        for i in range(epochs):
            print('Epoch : {}'.format(i+1))
            epoch_idx = (class_epoch if optimize_class else rank_epoch) + 1
            epoch_type = 'Class' if optimize_class else 'Rank'
            print('Optimize {} Epoch: {}'.format(epoch_type, epoch_idx))

            optimizer = cnn_optimizer if optimize_class else apn_optimizer
            optimizer.update_lr(epoch_idx)
            self.model.train()
            self.do_epoch(epoch_idx, optimizer, optimize_class=optimize_class)
            errors = self.eval()
            accuracy = 100 - errors[-1][0]  # Top-1 accuracy.
            error_history.append(errors)

            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'args': vars(self.args),
                }, fout)

            if optimize_class:
                class_epoch += 1
            else:
                rank_epoch += 1

            if (accuracy - best_accuracy) < self.args.converge_acc_diff:
                optimize_class = not optimize_class
                # self.model.flip_cnn_grads()
                # self.model.flip_apn_grads()
            # Save best model, if required.
            if accuracy > best_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_accuracy, accuracy))
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)

        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
        print('-' * 16)


class Optimizers(object):
    """Handles a list of optimizers."""

    def __init__(self, args):
        self.optimizers = []
        self.lrs = []
        self.decay_every = []
        self.args = args

    def add(self, optimizer, learning_rate, decay_every):
        """Adds optimizer to list."""
        self.optimizers.append(optimizer)
        self.lrs.append(learning_rate)
        self.decay_every.append(decay_every)

    def step(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.step()

    def update_lr(self, epoch_idx):
        """Update learning rate of every optimizer."""
        for optimizer, init_lr, decay_every in zip(self.optimizers, self.lrs, self.decay_every):
            optimizer = utils.step_lr(
                epoch_idx, init_lr, decay_every,
                self.args.lr_decay_factor, optimizer
            )


class ACC_loss(object):
    def __init__(self):
        self.acc = 0
        self.count = 0
        self.loss = 0

    def reset(self):
        self.acc = 0
        self.count = 0
        self.loss = 0

    def add_loss(self, loss):
        self.loss += loss.cpu().data.numpy()
        # self.count = loss.size(0)
        self.count += 1

    def print_loss(self, epoch, total_epoch):
        print("Epoch [%d/%d] Train Loss: %.2f " % (epoch+1, total_epoch, float(self.loss)/self.count))
        self.reset()


    def add_acc(self, correct, total):
        self.acc += correct
        self.count += total

    def print_acc(self):
        print("Test Accuracy: %.4f" % (100 * float(self.acc) / self.count))
        self.reset()




class Try(object):
    def __init__(self, net, train_dataiter, val_dataiter, lr, total_epoch):
        self.model=net
        self.model.cuda()
        self.train_dataiter = train_dataiter
        self.val_dataiter = val_dataiter
        self.lr = lr
        self.criterion=nn.CrossEntropyLoss()
        # self.optimizer=optim.SGD(self.model.parameters(
        # ), lr=lr, momentum=0.9, weight_decay=0.0001)
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=0.0001)
        self.total_epoch=total_epoch
        self.record=ACC_loss()

    def eval(self):
        self.model.eval()
        for batch, label in self.val_dataiter:
            batch=Variable(batch.cuda())
            labels=Variable(label.cuda())

            outputs=self.model(batch)
            _, predicted=torch.max(outputs.data, 1)

            total=labels.size(0)
            correct=(predicted == labels.data).sum()
            self.record.add_acc(correct, total)
        self.record.print_acc()


    def train(self):
        for i in range(self.total_epoch):
            self.do_epoch(i)
            self.record.print_loss(i, self.total_epoch)
            self.eval()
            if (((i+1) % int(self.total_epoch // 3)) == 0):
                self.lr /= 10
                print('reset learning rate to:', self.lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                    print(param_group['lr'])


    def do_epoch(self, i):
        self.model.train()
        for batch, label in tqdm(self.train_dataiter, desc='train [%d/%d]' % (i+1, self.total_epoch)):
            self.do_batch(batch, label)


    def do_batch(self, batch, label):
        batch=Variable(batch.cuda())
        label=Variable(label.cuda())

        self.optimizer.zero_grad()
        outputs=self.model(batch)
        loss=self.criterion(outputs, label)
        loss.backward()
        self.optimizer.step()

        self.record.add_loss(loss)


class Try_another(Try):
    def __init__(self, net, train_dataiter, val_dataiter, lr, total_epoch, index, optimizer=None):
        super(Try_another,self).__init__(net, train_dataiter, val_dataiter, lr, total_epoch)
        self.index = index
        if not optimizer == None:
            self.optimizer = optim.Adam(optimizer,lr=self.lr,weight_decay=0.0001)
        # self.params = list(net.attention1.trunk_branches.parameters())

    def optimizer_not(self):
        if self.params == list(net.attention1.trunk_branches.parameters()):
            print ('change')
            print (list(net.attention1.trunk_branches.parameters()))


    def train(self):
        for i in range(self.total_epoch):
            self.do_epoch(i)
            self.record.print_loss(i, self.total_epoch)
            self.eval()
            # self.optimizer_not()
            if (((i+1) % int(self.total_epoch // 3)) == 0):
                self.lr /= 10
                print('reset learning rate to:', self.lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                    print(param_group['lr'])

    def do_epoch(self,i):
        self.model.train()
        for batch, label in tqdm(self.train_dataiter, desc='train [%d/%d]' % (i+1, self.total_epoch)):
            self.do_batch(batch[self.index], label)

    def eval(self):
        self.model.eval()
        for batch, label in self.val_dataiter:
            batch = batch[self.index]
            batch=Variable(batch.cuda())
            labels=Variable(label.cuda())

            outputs=self.model(batch)
            _, predicted=torch.max(outputs.data, 1)

            total=labels.size(0)
            correct=(predicted == labels.data).sum()
            self.record.add_acc(correct, total)
        self.record.print_acc()



class Save_features(object):
    def __init__(self, net, train_dataiter, val_dataiter):
        self.model=net.eval()
        self.model.cuda()
        # self.model.train()
        self.features=[]
        self.labels=[]
        for batch, label in tqdm(train_dataiter, desc='save'):
            self.do_batch(batch, label)

        self.features=torch.cat(self.features, dim=0)
        self.labels=torch.cat(self.labels, dim=0)

        ckpt={
            'train_features': self.features,
            'train_labels': self.labels,
        }
        self.features=[]
        self.labels=[]
        for batch, label in tqdm(val_dataiter, desc='save'):
            self.do_batch(batch, label)

        self.features=torch.cat(self.features, dim=0)
        self.labels=torch.cat(self.labels, dim=0)
        ckpt['val_features']=self.features
        ckpt['val_labels']=self.labels

        # Save to file.
        torch.save(ckpt, 'features_labels.pt')


    def do_batch(self, batch, label):
        feature=[]
        for b in batch:
            b=b.cuda()
            b=Variable(b)
            result=self.model(b)

            result=result.cpu().data
            feature.append(result)
        feature=torch.cat(feature, dim=1)

        self.features.append(feature)
        self.labels.append(label)

from model.residual_attention_network import ResidualAttentionModel_BU3DFE_CAT as ResidualAttentionModel
from model.residual_attention_network import ResidualAttentionModel_92 as Attention
from model.residual_attention_network import VGGAttention

if __name__ == '__main__':
    ###########保存 feature
     # net = ResidualAttentionModel()
     # # net = VGG16_bn(1)
     # train_dataiter,val_dataiter = loader_bu3dfe('../data/BU3DFE-2D/BU3DFE.txt','/home/muyouhang/zkk/BU3DFE/data/BU3DFE-2D',10,shuffle=False,image=32)
     # Save_features(net,train_dataiter,val_dataiter)

     #############  feature 训练
    # dataiter=torch.load('features_labels.pt')
    # train_dataiter=loader_tensor(
    #     dataiter['train_features'], dataiter['train_labels'], batch_size=10)
    # val_dataiter=loader_tensor(
    #     dataiter['val_features'], dataiter['val_labels'], batch_size=10)

    # net=Classifier(6144, 6)

    # contain=Try(net, train_dataiter, val_dataiter, 0.001, 90)
    # contain.train()

    ############ end to end 训练attention 模型  选择一张图
    # net = Attention(6)
    # train_dataiter,val_dataiter = loader_bu3dfe('../data/BU3DFE-2D/BU3DFE.txt','/home/muyouhang/zkk/BU3DFE/data/BU3DFE-2D',10,shuffle=False,image=224)
    # contain=Try_another(net, train_dataiter, val_dataiter, 0.00001, 90, 0)
    # contain.train()

    ############## 只训练attention模型 的 attention部分
    # net = VGGAttention()
    # train_dataiter,val_dataiter = loader_bu3dfe('../data/BU3DFE-2D/BU3DFE.txt','/home/muyouhang/zkk/BU3DFE/data/BU3DFE-2D',16,shuffle=True,image=112)
    # optimizer = list(net.attention1.parameters()) + list(net.attention2.parameters()) + list(net.attention3.parameters()) + list(net.attention4.parameters()) + list(net.classifier.parameters())
    # contain=Try_another(net, train_dataiter, val_dataiter, 0.001, 15, 4,optimizer=optimizer)
    # contain.train()

    ############# 只训练VGG# net的calssifier

    #net = VGG16_bn(6)
    net = VGG16_bn_finetuned(6)
    train_dataiter,val_dataiter = loader_bu3dfe('../data/BU3DFE-2D/BU3DFE.txt','/home/muyouhang/zkk/BU3DFE/data/BU3DFE-2D',16,shuffle=True,image=224)
    optimizer = list(net.classifer.parameters())
    contain=Try_another(net, train_dataiter, val_dataiter, 0.001, 15, 4,optimizer=optimizer)
    contain.train()




