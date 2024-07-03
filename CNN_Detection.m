clc
clear all
categories = {'AG','BG'};
rootFolder = 'MGFTrain';
imds = imageDatastore(fullfile(rootFolder, categories),'LabelSource','foldernames');
%imageAugmenter = imageDataAugmenter('RandRotation',[-180 180]);
%imd_train=augmentedImageSource([15 15 1],imds,'DataAugmentation',imageAugmenter);

%% Define Layers

layers = [
    imageInputLayer([15 15 1]);
    convolution2dLayer(5,32,'Padding',0,'Stride',1,'WeightLearnRateFactor',1);
    maxPooling2dLayer(3,'Padding',1,'Stride', 2);
    batchNormalizationLayer();
    reluLayer();
    convolution2dLayer(5,48,'Padding',2,'Stride',1,'WeightLearnRateFactor',1);
    averagePooling2dLayer(3,'Padding',2,'Stride', 1);
    batchNormalizationLayer();
    reluLayer(); 
    convolution2dLayer(5,24,'Padding',2,'Stride',1,'WeightLearnRateFactor',1);
    averagePooling2dLayer(3,'Padding',2,'Stride', 1);
    batchNormalizationLayer();
    reluLayer();
    fullyConnectedLayer(64,'BiasLearnRateFactor',1);
    reluLayer();
    dropoutLayer;
    fullyConnectedLayer(2,'BiasLearnRateFactor',1);
    softmaxLayer();
    classificationLayer()];

%% Training Options
 opts=trainingOptions('sgdm',...
      'LearnRateSchedule','piecewise',...
      'InitialLearnRate',0.001,...
      'MaxEpochs',5,... 
      'MiniBatchSize',128,...
      'Verbose', true,...
      'VerboseFrequency',10,...
      'Plots','training-progress',...
      'ExecutionEnvironment','auto',...
      'OutputFcn',@(info)stopIfAccuracyNotImproving(info,5));
%% Train the Network

[MGFnet, info] = trainNetwork(imds,layers,opts)
%{

confMat=confusionmat(imds_test.Labels,Mgtest);
confMat=confMat./sum(confMat,2)
mean(diag(confMat))
%}