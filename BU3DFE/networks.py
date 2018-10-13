import torch
import numpy as np
import torch.nn as nn
from torchvision import models

IMSIZE = 224
'''
Public facing networks
'''
class APN2(nn.Module):
    '''
    This networks wraps two APNs for APN pretraining
    '''
    def __init__(self, num_classes, cnn):
        super(APN2, self).__init__()
        self.cnn = cnn(num_classes, IMSIZE)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.apn1 = APN(self.cnn.n_features)
        self.apn2 = APN(self.cnn.n_features)
        self.cropup = CropUpscale((IMSIZE, IMSIZE))

    def forward(self, x, crop_params=None):
        h = x.size(2)
        _, feats = self.cnn(x)
        crop_params1 = self.apn1(feats)
        if crop_params is not None:
            cx, cy, hw = int(h*crop_params[0, 0]), int(h*crop_params[0, 1]), int(h*crop_params[0, 2])//2
            crop_x = x[:, :, cx-hw:cx+hw, cy-hw:cy+hw]
            crop_x = nn.Upsample(size=(h, h), mode='bilinear')(crop_x)
        else:
            crop_x = self.cropup(x, crop_params1*h)
        _, feats = self.cnn(crop_x)
        crop_params2 = self.apn2(feats)
        return torch.cat([crop_params1, crop_params2], 1)




"""Private Layers and Networks"""

class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class VGG(nn.Module):
    """
    VGG16 with Fine Grained Classification Head
    """
    def __init__(self, num_classes, im_size):
        super(VGG, self).__init__()
        assert(im_size == 224 or im_size == 448)
        pool_size = 14 if im_size == 224 else 28

        base_model = models.vgg19(pretrained=True)
        base_features = list(base_model.features)
        self.features = [*base_features[:-2]]
        # print (base_features[:-2])
        self.n_features = 512 * pool_size * pool_size
        self.flatten_features = View(-1, self.n_features)
        self.features = nn.Sequential(*self.features)
        base_classifier = list(base_model.classifier.children())
        # print (base_classifier)
        fc6 = nn.Linear(512 * pool_size//2 * pool_size//2, 4096) if im_size == 448 else base_classifier[0]
        self.classifier = nn.Sequential(
                *base_features[-2:],
                View(-1, 512 * pool_size//2 * pool_size//2),
                fc6,
                *base_classifier[1:-1],
                nn.Linear(4096, num_classes)
        )
        for mod in self.classifier:
            if isinstance(mod, nn.ReLU):
                mod.inplace = False

    def forward(self, x, flatten=True):
        """
        Applies VGG16 forward pass for class wise scores
        :param input: (num_batch, 3, h, w) np array batch of images to find class wise scores of
        :return: (num_batch, num_classes) np array of class wise scores per image
        """
        feats = self.features(x)
        out = self.classifier(feats)
        if flatten:
            feats = self.flatten_features(feats)

        return out, feats

class Inception_v3(nn.Module):
    """docstring for vgg"""
    def __init__(self, num_classes):
        super(Inception_v3, self).__init__()
        # self.n_features =


        base_model = models.inception_v3(pretrained=True)

        base_features = list(base_model.features)
        self.features = [*base_features[:]]
        self.features = nn.Sequential(*self.features)
        # self.flatten_features = View(-1, self.n_features)
    def forward(self,x):

        feats = self.features(x)

        return feats


class Classifier(nn.Module):
    def __init__(self,input_shape,num_classes):
        super(Classifier,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_shape,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
            )
    def forward(self,x):
        return self.classifier(x)

class VGG16_bn_finetuned(nn.Module):
    """docstring for vgg16_bn"""
    def __init__(self, num_classes):
        super(VGG16_bn_finetuned, self).__init__()
        base_model = models.vgg16_bn(pretrained=True)

        base_features = list(base_model.features)

        self.features = [*base_features[:]]

        self.flatten_features = View(-1, 512*7*7)
        self.features = nn.Sequential(*self.features,self.flatten_features)
        base_classifer = list(base_model.classifier)
        self.classifer = nn.Sequential(*base_classifer[:-1],nn.Linear(4096,num_classes))
    def forward(self,x):

        feats = self.features(x)

        out = self.classifer(feats)
        return out



class VGG16_bn(nn.Module):
    """docstring for vgg16_bn"""
    def __init__(self, num_classes):
        super(VGG16_bn, self).__init__()
        base_model = models.vgg19(pretrained=True)

        base_features = list(base_model.features)

        self.features = [*base_features[:]]

        self.flatten_features = View(-1, 512)
        self.features = nn.Sequential(*self.features,
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1),
            self.flatten_features)

        self.classifer = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128,num_classes)
        )


    def forward(self,x):

        feats = self.features(x)

        out = self.classifer(feats)
        return out



if __name__ == '__main__':
    net = VGG(200,224)
