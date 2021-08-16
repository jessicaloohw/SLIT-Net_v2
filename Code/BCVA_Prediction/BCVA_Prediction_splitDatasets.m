%
% SLIT-Net v2
% DOI: TBC
%
% Split BCVA prediction dataset into 6-fold
%

clear;
close all;
clc;

% User-input:
datasetFolder = '../../Datasets/BCVA_Prediction';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get paths:
pathToDataset = fullfile(datasetFolder, 'measurements.mat');
pathToIdxs = fullfile(datasetFolder, 'idxs.mat');

% Get appropriate indices:
load(pathToIdxs); % Load variables k1 - k6

for k = 1:6
    
    disp(['K',num2str(k)]);
    
    switch k
        case 1
            trainIdxs = [k2, k3, k4, k5, k6];
            testIdxs = k1;
            saveTag = 'K1';
        case 2
            trainIdxs = [k3, k4, k5, k6, k1];
            testIdxs = k2;
            saveTag = 'K2';
        case 3
            trainIdxs = [k4, k5, k6, k1, k2];
            testIdxs = k3;
            saveTag = 'K3';
        case 4
            trainIdxs = [k5, k6, k1, k2, k3];
            testIdxs = k4;
            saveTag = 'K4';
        case 5
            trainIdxs = [k6, k1, k2, k3, k4];
            testIdxs = k5;
            saveTag = 'K5';
        case 6
            trainIdxs = [k1, k2, k3, k4, k5];
            testIdxs = k6;
            saveTag = 'K6';
    end
    
    % Sort:
    trainIdxs = sort(trainIdxs);
    testIdxs = sort(testIdxs);
    
    splitDatasets_withIndices(pathToDataset, trainIdxs, testIdxs, datasetFolder, saveTag);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [] = splitDatasets_withIndices(pathToDataset, trainIdxs, testIdxs, saveFolder, saveTag)

% Load data:
allData = load(pathToDataset);
allData = allData.data;
nData = length(allData);
disp(['Dataset: ',num2str(nData),'measurements.']);

% Save training data:
if(~isempty(trainIdxs))
    data = allData(trainIdxs);
    save(fullfile(saveFolder,['train_data_',saveTag,'.mat']), '-v7.3', 'data');
    disp(['Training: ',num2str(length(data)),' measurements.']);
end

% Save testing data:
if(~isempty(testIdxs))
    data = allData(testIdxs);
    save(fullfile(saveFolder,['test_data_',saveTag,'.mat']), '-v7.3', 'data');
    disp(['Testing: ',num2str(length(data)),' measurements.']);
end

disp('Dataset split completed.');

end