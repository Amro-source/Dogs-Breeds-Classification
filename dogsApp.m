

clc; clear all; close all;




imds = imageDatastore('dogbreeds', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% Divide the data into training and validation data sets. Use 70% of the images for training and 30% for validation. splitEachLabel splits the images datastore into two new datastores.
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
% This very small data set now contains 55 training images and 20 validation images. Display some sample images.
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end



% Load Pretrained Network
% Load the pretrained AlexNet neural network. If Deep Learning Toolbox™ Model for AlexNet Network is not installed, then the software provides a download link. AlexNet is trained on more than one million images and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the model has learned rich feature representations for a wide range of images.
net = alexnet;
% Use analyzeNetwork to display an interactive visualization of the network architecture and detailed information about the network layers.
analyzeNetwork(net)


inputSize = net.Layers(1).InputSize
% inputSize = 1×3
% 
%    227   227     3



