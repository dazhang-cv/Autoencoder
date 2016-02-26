%clc; clear all; close all;

% learning separate region representations

% set the training parameters
inputSize = 32*32;

pack1_neg_Size = 12940;

%pack1_neg_Data = zeros(inputSize, pack1_neg_Size);

%fprintf('# loading pack1 neg vehicle images(size): %d\n', pack1_neg_Size);
%for i = 0:pack1_neg_Size-1
%    image_name = strcat('../pack1/falsetemp/',num2str(i),'.bmp');
%    img = imread(image_name);
%    img2 = im2double(img);
%    pack1_neg_Data(:,i+1) = reshape(img2,[inputSize,1]);
%end
%fprintf('# Load pack1 neg set complete \n');

idx = randperm(12940,4000);

for i = 1:4000
    image = reshape(pack1_neg_Data(:,idx(i)),[32,32]);
    imwrite(image,sprintf('%s/%d.png','C:\KITTI_Dataset\training\Car\train\neg',i));
end