function result = SAE_classify(input)

% input: a 32*32 image patch, color or gray scale
% result: classification result, 1 for vehicle and 2 for non-vehicle

% load the neural network parameters
inputSize = 32*32;
featureSize = 512;
numLabels = 2;

load stackedAEOptTheta.mat;
load netconfig.mat;

% test the image
[width, height, channel] = size(input);
% grayscale image
if channel == 1
    img = input;
end
% color image
if channel == 3
    img = rgb2gray(input);
end
img = im2double(img);
test = reshape(img,[inputSize,1]);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, featureSize, ...
                          numLabels, netconfig, test);
                      
result = pred(1);

end