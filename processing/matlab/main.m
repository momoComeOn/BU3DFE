clc
clear 

data = get_wrl_xyz("E:\\���ݿ�\\BU-3DFE\\BU-3DFE\\F0001\\F0001_AN01WH_F3D.wrl");

img = get_img_new(data);
img_norm = get_img_norm(img);

% a = medfilt2(img_norm(:,:,2));
imshow(rot90(img_norm(:,:,1)));

img = rot90(img_norm(:,:,1));

imwrite(img,'img.jpg')


% img_ = get_img(data);
% 
% img_ = ordfilt2(img_,9,ones(3,3))
% img_norm_ = get_img_norm(img_);
% 
% 
% figure;
% imshow(img_norm_(:,:,1))


% figure;
% img = load("result.mat");
% data = [double(img.vertices(:,1)),double(img.vertices(:,2)),img.shape_index(1,:)'];
% img =  get_img_new(data);
% imshow(rot90(img(:,:)));

% x=[];
% y=[];
% for i = 1:188
%     y = [y 1:149];
%     x = [x ones(1,149)*i];
% end
% 
% depth = reshape(img,1,28012);
% 
% tri = delaunay(x(1,:), y(1,:));
% 
% str = struct('ver',{[x(1,:)',y(1,:)',depth(1,:)']},'face',{tri});
% save('reason.mat','str');


