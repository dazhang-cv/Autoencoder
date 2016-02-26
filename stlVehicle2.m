clc; clear all; close all;

% learning separate region representations

% set the training parameters
inputSize = 32*32;
numLabels = 2;
hiddenSize = 256;
sparsityParam = 0.1;

pack1_pos_Size = 6093;
pack1_neg_Size = 12940;

pack2_pos_Size = 5330;
pack2_neg_Size = 5690;


lambda = 3e-3;
beta = 3;
maxIter = 400;
% load vehicle data from file

patches = zeros(inputSize,pack1_pos_Size);

fprintf('# loading pack1 pos vehicle images(size): %d\n', pack1_pos_Size);
for i = 0:pack1_pos_Size-1
    image_name = strcat('../pack1/truetemp/',num2str(i),'.bmp');
    img = imread(image_name);
    img2 = im2double(img);
    patches(:,i+1) = reshape(img2,[inputSize,1]);
end
fprintf('# Load pack1 pos set complete \n');

pack1_neg_Data = zeros(inputSize, pack1_neg_Size);

fprintf('# loading pack1 neg vehicle images(size): %d\n', pack1_neg_Size);
for i = 0:pack1_neg_Size-1
    image_name = strcat('../pack1/falsetemp/',num2str(i),'.bmp');
    img = imread(image_name);
    img2 = im2double(img);
    pack1_neg_Data(:,i+1) = reshape(img2,[inputSize,1]);
end
fprintf('# Load pack1 neg set complete \n');

testData = zeros(inputSize, pack2_pos_Size + pack2_neg_Size);

fprintf('# loading pack2 vehicle images(size): %d\n', pack2_pos_Size + pack2_neg_Size);
for i = 0:pack2_pos_Size-1
    image_name = strcat('../pack2/truetemp/',num2str(i),'.bmp');
    img = imread(image_name);
    img2 = im2double(img);
    testData(:,i+1) = reshape(img2,[inputSize,1]);
end
for i = 0:pack2_neg_Size-1
    image_name = strcat('../pack2/falsetemp/',num2str(i),'.bmp');
    img = imread(image_name);
    img2 = im2double(img);
    testData(:,pack2_pos_Size+i+1) = reshape(img2,[inputSize,1]);
end
fprintf('# Load pack1 neg set complete \n');


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

% softmax regression classifier

% use pack1 data to train the classifier
% use pack2 data to test the classifier
trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       [patches pack1_neg_Data]);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);
                                   
% manually set the training and testing labels
trainLabels = [ones(1,pack1_pos_Size) zeros(1,pack1_neg_Size)+2];
testLabels = [ones(1,pack2_pos_Size) zeros(1,pack2_neg_Size)+2];
                                   
% train the softmax classifier

softmaxModel = struct;  
lambda = 1e-4; 
options.maxIter = 100;

softmaxModel = softmaxTrain(hiddenSize, numLabels, lambda, ...
                            trainFeatures, trainLabels, options);

% softmax prediction                        
[pred] = softmaxPredict(softmaxModel, testFeatures);

% classification score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));
fprintf('False Positive: %f%%\n', 100-100*mean(pred(pack2_pos_Size+1:end) == testLabels(pack2_pos_Size+1:end)));
fprintf('False Negative: %f%%\n', 100-100*mean(pred(1:pack2_pos_Size) == testLabels(1:pack2_pos_Size)));