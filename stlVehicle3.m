clc; clear all; close all;

% separate the image into 4 sub-regions and try to learn the representation
% for each sector.

% set the training parameters
inputSize = 32*32;
numLabels = 2;
hiddenSize = 9;
sparsityParam = 0.1;

trainSize = 6093;
testSize = 12940;

lambda = 3e-3;
beta = 3;
maxIter = 400;
% load vehicle data from file

inputSize = inputSize/4;

patches = zeros(inputSize,trainSize*4);

fprintf('# loading true vehicle images(size): %d\n', trainSize);
for i = 0:trainSize-1
    image_name = strcat('../pack1/truetemp/',num2str(i),'.bmp');
    img = imread(image_name);
    img2 = im2double(img);
    A = img2(1:16,1:16);
    B = img2(1:16,17:32);
    C = img2(17:32,1:16);
    D = img2(17:32,17:32);
    patches(:,4*i+1) = reshape(A,[inputSize,1]);
    patches(:,4*i+2) = reshape(B,[inputSize,1]);
    patches(:,4*i+3) = reshape(C,[inputSize,1]);
    patches(:,4*i+4) = reshape(D,[inputSize,1]);
end
fprintf('# Load training set complete \n');


% training autoencoder
% randomly set the initial parameter
theta = initializeParameters(hiddenSize, inputSize);
opttheta = zeros(size(theta));

addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

options.Corr = 100;
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                                   theta, options);

% display the trained network                              
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');
