%
% SLIT-Net v2
% DOI: TBC
%
% Split segmentation dataset into 6-fold
%

clear;
close all;
clc;

% User-input:
segmentationFolder = '../../Datasets/Segmentation';
lightType = 'White'; % 'White', 'Blue'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get dataset folder:
datasetFolder = fullfile(segmentationFolder, [lightType,'_Light']);

% Get paths:
pathToDataset = fullfile(datasetFolder, 'annotations.mat');
pathToIdxs = fullfile(datasetFolder, 'idxs.mat');

% Get appropriate indices:
load(pathToIdxs); % Load variables k1 - k6

for k = 1:6
    
    disp(['K',num2str(k)]);
    
    switch k
        case 1
            trainIdxs = [k2, k3, k4, k5];
            valIdxs = k6;
            testIdxs = k1;
            saveTag = 'K1';
        case 2
            trainIdxs = [k3, k4, k5, k6];
            valIdxs = k1;
            testIdxs = k2;
            saveTag = 'K2';
        case 3
            trainIdxs = [k4, k5, k6, k1];
            valIdxs = k2;
            testIdxs = k3;
            saveTag = 'K3';
        case 4
            trainIdxs = [k5, k6, k1, k2];
            valIdxs = k3;
            testIdxs = k4;
            saveTag = 'K4';
        case 5
            trainIdxs = [k6, k1, k2, k3];
            valIdxs = k4;
            testIdxs = k5;
            saveTag = 'K5';
        case 6
            trainIdxs = [k1, k2, k3, k4];
            valIdxs = k5;
            testIdxs = k6;
            saveTag = 'K6';
    end
    
    % Sort:
    trainIdxs = sort(trainIdxs);
    valIdxs = sort(valIdxs);
    testIdxs = sort(testIdxs);
    
    splitDatasets_withIndices(pathToDataset, trainIdxs, valIdxs, testIdxs, datasetFolder, saveTag);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [] = splitDatasets_withIndices(pathToDataset, trainIdxs, valIdxs, testIdxs, saveFolder, saveTag)

% Load data:
allData = load(pathToDataset);
allData = allData.data;
nData = length(allData);
disp(['Dataset: ',num2str(nData),' images.']);

% Save training data:
if(~isempty(trainIdxs))
    data = allData(trainIdxs);
    save(fullfile(saveFolder,['train_data_',saveTag,'.mat']), '-v7.3', 'data');
    disp(['Training: ',num2str(length(data)),' images.']);
end

% Save validation data:
if(~isempty(valIdxs))
    data = allData(valIdxs);
    save(fullfile(saveFolder,['val_data_',saveTag,'.mat']), '-v7.3', 'data');
    disp(['Validation: ',num2str(length(data)),' images.']);
end

% Save testing data:
if(~isempty(testIdxs))
    data = allData(testIdxs);
    save(fullfile(saveFolder,['test_data_',saveTag,'.mat']), '-v7.3', 'data');
    disp(['Testing: ',num2str(length(data)),' images.']);
end

disp('Dataset split completed.');

end