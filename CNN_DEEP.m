clc
clear all
%% Load training data
categories = {'AG','BG','CG','AB','AC','BC','ABG','ACG','BCG','ABCG','Nofault'};
rootFolder = 'MGFTrain';
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
%imageAugmenter = imageDataAugmenter('RandRotation',[0 360]);
%Imds_train=augmentedImageSource([24 24 1],imds,'DataAugmentation',imageAugmenter);
%{
%% Define Layers
    layers = [
    imageInputLayer([24 24 1]);
    convolution2dLayer(5,32,'Padding',2,'Stride',1);
    maxPooling2dLayer(2,'Stride', 2);
    batchNormalizationLayer();
    reluLayer();
    convolution2dLayer(5,64,'Padding',2,'Stride',1);
    averagePooling2dLayer(2,'Stride', 2);
    batchNormalizationLayer();
    reluLayer();
    convolution2dLayer(3,64,'Padding',0,'Stride',1);
    averagePooling2dLayer(2,'Stride', 2);
    batchNormalizationLayer();
    reluLayer(); 
    convolution2dLayer(3,128,'Padding',1,'Stride',1);
    maxPooling2dLayer(2,'Stride', 2);
    fullyConnectedLayer(128);
    reluLayer();
    dropoutLayer;
    fullyConnectedLayer(11);
    softmaxLayer();
    classificationLayer()];
 %% Training Options
 opts=trainingOptions('sgdm',...
      'InitialLearnRate',0.008,...
      'MaxEpochs',300,... 
      'MiniBatchSize',256,...
      'Verbose', true,...
      'VerboseFrequency',10,...
      'Plots','training-progress',...
      'ExecutionEnvironment','auto',...
      'OutputFcn',@(info)stopIfAccuracyNotImproving(info,5));
 [MGrFnet, info1] = trainNetwork(Imds_train,layers,opts)
%}