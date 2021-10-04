%
% SLIT-Net v2
% DOI: 10.1167/tvst.10.12.2
%
% Update test datasets with predicted limbus
%

clear;
close all;
clc;

% User-input:
mainFolder = '../../Datasets/Segmentation';
modelFolder = '../../Models/Segmentation/Limbus-Net';
modelNum = 300;

lightType = 'Blue';    % 'White', 'Blue'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dataset folder:
datasetFolder = fullfile(mainFolder,[lightType,'_Light']);

for k = 1:6
    
    % Dataset filename:
    pathToDataset = fullfile(datasetFolder,['test_data_K',num2str(k),'.mat']);
    
    % Load data:
    load(pathToDataset);
    nData = length(data);
    
    % Predicted masks folder:
    maskFolder = fullfile(modelFolder,[lightType,'_Light'],...
        ['K',num2str(k)],...
        'testing',...
        ['model-',num2str(modelNum)]);
    
    maskList = dir(fullfile(maskFolder,'*.mat'));
    maskList = {maskList.name};
    nMasks = length(maskList);
    
    assert(nData == nMasks);
    
    for i = 1:nData
        
        % Load predicted mask:
        pathToMask = fullfile(maskFolder, sprintf('%03d.mat',i-1));
        
        mask = load(pathToMask);
        mask = mask.predicted_mask;        
        
        % Get bounding box [x, y, w, h]
        [ys, xs] = find(mask);
        box = [min(xs)-1 min(ys)-1 max(xs)-min(xs)+1 max(ys)-min(ys)+1];
        
        % Update:
        data(i).PREDICTED_LIMBUS_MASK = mask;
        data(i).PREDICTED_LIMBUS_BOX = box; 
        
    end
    
    % Save:
    save(pathToDataset,'-v7.3','data');
    disp(['K',num2str(k),' | Dataset updated with predicted limbus.']);
    
end
