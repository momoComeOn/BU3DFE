clc
clear

file_folder = fullfile('E:\\数据库\\BU-3DFE\\BU-3DFE\\');

dir_folder = dir(fullfile(file_folder,'*','*D.wrl'));

h = strcat(cell2mat({dir_folder.folder}'),'\',cell2mat({dir_folder.name}'));
root = 'E:\数据库\BU3DFE-2D\';
destination = fullfile(root,'BU3DFE-NormalZ');

if ~exist(destination,'dir')
    mkdir(destination);
end
for i=1:length(h)

    
    dst = h(i,:);
    data = get_wrl_xyz(dst);
    img = get_img_new(data);
    img_norm = get_img_norm(img);
    img_D = rot90(img_norm(:,:,4));
    
    a = split(dst,["\","."]);
    des = [a{length(a)-1} '.bmp'];

    des = fullfile(destination,des);
    
    imwrite(img_D,des);
end




