clc; clear all; close all;

% only use pack1 to learn the representations and self-test using positive
% and negative datasets

% set the training parameters
inputSize = 32*32;
numLabels = 2;
hiddenSize = 100;
sparsityParam = 0.1;

trainSize = 6093;
testSize = 12940;

lambda = 3e-3;
beta = 3;
maxIter = 400;
% load vehicle data from file

patches = zeros(inputSize,trainSize);

fprintf('# loading true vehicle images(size): %d\n', trainSize);
for i = 0:trainSize-1
    image_name = strcat('../truetemp/',num2str(i),'.bmp');
    img = imread(image_name);
    img2 = im2double(img);
    patches(:,i+1) = reshape(img2,[inputSize,1]);
end
fprintf('# Load training set complete \n');

testData = zeros(inputSize, testSize);

fprintf('# loading false vehicle images(size): %d\n', testSize);
for i = 0:testSize-1
    image_name = strcat('../falsetemp/',num2str(i),'.bmp');
    img = imread(image_name);
    img2 = im2double(img);
    testData(:,i+1) = reshape(img2,[inputSize,1]);
end
fprintf('# Load testing set complete \n');


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

% use the truetemp data to train the classifier
trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       patches);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);
                                   
% manually set the training and testing labels
trainLabels = ones(1,trainSize);
testLabels = zeros(1,testSize) + 2;
                                   
% train the softmax classifier

softmaxModel = struct;  
lambda = 1e-4; 
options.maxIter = 100;

softmaxModel = softmaxTrain(hiddenSize, numLabels, lambda, ...
                            [trainFeatures testFeatures], [trainLabels testLabels], options);

% softmax prediction                        
[pred] = softmaxPredict(softmaxModel, testFeatures);

% classification score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% softmax prediction                        
[pred] = softmaxPredict(softmaxModel, trainFeatures);

% classification score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == trainLabels(:)));