%
% SLIT-Net v2
% DOI: TBC
%
% Update BCVA prediction datasetwith measurements
%

clear;
close all;
clc;

% User-input:
datasetFolder = '../../Datasets/BCVA_Prediction';
SLITNetSegmentationFolder = '../../Trained_Models/Segmentation/SLIT-Net_v2';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% IDs of interest:
whiteLimbusID = 7;
blueLimbusID = 4;

whiteRoiIDs = [1, 2, 3, 4];
blueRoiIDs = [1];

nWhiteROIs = length(whiteRoiIDs);
nBlueROIs = length(blueRoiIDs);

% Limbus length:
mmLimbusLength = 11.7;  % mm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get segmentation folders:
whiteSegmentationFolder = fullfile(SLITNetSegmentationFolder,'White_Light','Final_Segmentations');
blueSegmentationFolder = fullfile(SLITNetSegmentationFolder,'Blue_Light','Final_Segmentations');

% Get path:
pathToDataset = fullfile(datasetFolder, 'measurements.mat');

% Load data:
load(pathToDataset);
nData = length(data);

%%%%%%%%%%%%%%%%% UPDATE DATASET WITH MEASUREMENTS %%%%%%%%%%%%%%%%%%%%%%%%


for i = 1:nData
    
    disp(['DATA ',num2str(i)]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%% WHITE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Initialize measurements:
    whiteMeasurements = zeros([nWhiteROIs, 5]);
    
    % Get VXXX:
    [~, whiteVXXX, ~] = fileparts(data(i).WHITE_PATH);
    
    % Get segmentations (class_ids, masks, rois, scores):
    whiteSegmentationFilename = fullfile(whiteSegmentationFolder, whiteVXXX, 'segmentations.mat');
    whiteSegmentations = load(whiteSegmentationFilename);
    whiteLabels = whiteSegmentations.class_ids;
    whiteMasks = whiteSegmentations.masks;
    
    % Get limbus:
    whiteLimbusMask = [];
    
    for j = 1:length(whiteLabels)
        if whiteLabels(j) == whiteLimbusID
            if isempty(whiteLimbusMask)
                whiteLimbusMask = whiteMasks(:,:,j);
            end
        end
    end
    
    if isempty(whiteLimbusMask)
        whiteLimbusLength = nan;
        whiteLimbusArea = nan;
        whiteLimbusCenter = nan;
    else
        [whiteLimbusLength, whiteLimbusArea, whiteLimbusCenter] = getMeasurements(whiteLimbusMask);
        if length(whiteLimbusLength) > 1
            [whiteLimbusLength, whiteLimbusMaxIdx] = max(whiteLimbusLength);
            whiteLimbusCenter = whiteLimbusCenter(whiteLimbusMaxIdx, :);
        end
    end
    
    % Get pixel pitch:
    whitePixelPitch = mmLimbusLength / whiteLimbusLength;
    
    for r = 1:nWhiteROIs
        
        % Initialise:
        whiteMask_r = zeros([size(whiteMasks,1), size(whiteMasks,2)]);
        
        % Combine all of the same IDs:
        for j = 1:length(whiteLabels)
            if whiteLabels(j) == whiteRoiIDs(r)
                whiteMask_r = whiteMask_r + whiteMasks(:,:,j);
            end
        end
        whiteMask_r = (whiteMask_r > 0);
        
        % Presence:
        if sum(whiteMask_r(:)) > 0
            whitePresence_r = 1;
        else
            whitePresence_r = 0;
        end
        
        if whitePresence_r
            
            % Get measurements:
            [whiteWidth_r, whiteArea_r, whiteCenter_r] = getMeasurements(whiteMask_r);
            
            % Maximum width:
            whiteWidth_r = max(whiteWidth_r);
            whiteWidth_r = whiteWidth_r * whitePixelPitch;
            
            % Area:
            whiteArea_r = whiteArea_r * whitePixelPitch * whitePixelPitch;
            
            % Percentage area:
            whitePercentage_r = (whiteArea_r / (whiteLimbusArea * whitePixelPitch * whitePixelPitch));
            
            % Centrality:
            whiteDistance_r = (whiteCenter_r - whiteLimbusCenter) * whitePixelPitch;
            whiteDistance_r = sqrt(sum(whiteDistance_r .^ 2, 2));
            whiteDistance_r = min(whiteDistance_r);
            whiteCentrality_r = log(1 + (1/whiteDistance_r));
            
        else
            
            whiteWidth_r = 0;
            whiteArea_r = 0;
            whitePercentage_r = 0;
            whiteCentrality_r = 0;
            
        end
        
        % Update measurements:
        whiteMeasurements(r, 1) = whitePresence_r;
        whiteMeasurements(r, 2) = whiteWidth_r;
        whiteMeasurements(r, 3) = whiteArea_r;
        whiteMeasurements(r, 4) = whitePercentage_r;
        whiteMeasurements(r, 5) = whiteCentrality_r;
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% BLUE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Initialize measurements:
    blueMeasurements = zeros([nBlueROIs, 5]);
    
    % Get VXXX:
    [~, blueVXXX, ~] = fileparts(data(i).BLUE_PATH);
    
    % Get segmentations (class_ids, masks, rois, scores):
    blueSegmentationFilename = fullfile(blueSegmentationFolder, blueVXXX, 'segmentations.mat');
    blueSegmentations = load(blueSegmentationFilename);
    blueLabels = blueSegmentations.class_ids;
    blueMasks = blueSegmentations.masks;
    
    % Get limbus:
    blueLimbusMask = [];
    
    for j = 1:length(blueLabels)
        if blueLabels(j) == blueLimbusID
            if isempty(blueLimbusMask)
                blueLimbusMask = blueMasks(:,:,j);
            end
        end
    end
    
    if isempty(blueLimbusMask)
        blueLimbusLength = nan;
        blueLimbusArea = nan;
        blueLimbusCenter = nan;
    else
        [blueLimbusLength, blueLimbusArea, blueLimbusCenter] = getMeasurements(blueLimbusMask);
        if length(blueLimbusLength) > 1
            [blueLimbusLength, blueLimbusMaxIdx] = max(blueLimbusLength);
            blueLimbusCenter = blueLimbusCenter(blueLimbusMaxIdx, :);
        end
    end
    
    % Get pixel pitch:
    bluePixelPitch = mmLimbusLength / blueLimbusLength;
    
    for r = 1:nBlueROIs
        
        % Initialise:
        blueMask_r = zeros([size(blueMasks,1), size(blueMasks,2)]);
        
        % Combine all of the same IDs:
        for j = 1:length(blueLabels)
            if blueLabels(j) == blueRoiIDs(r)
                blueMask_r = blueMask_r + blueMasks(:,:,j);
            end
        end
        blueMask_r = (blueMask_r > 0);
        
        % Presence:
        if sum(blueMask_r(:)) > 0
            bluePresence_r = 1;
        else
            bluePresence_r = 0;
        end
        
        if bluePresence_r
            
            % Get measurements:
            [blueWidth_r, blueArea_r, blueCenter_r] = getMeasurements(blueMask_r);
            
            % Maximum width:
            blueWidth_r = max(blueWidth_r);
            blueWidth_r = blueWidth_r * bluePixelPitch;
            
            % Area:
            blueArea_r = blueArea_r * bluePixelPitch * bluePixelPitch;
            
            % Percentage area:
            bluePercentage_r = (blueArea_r / (blueLimbusArea * bluePixelPitch * bluePixelPitch));
            
            % Centrality:
            blueDistance_r = (blueCenter_r - blueLimbusCenter) * bluePixelPitch;
            blueDistance_r = sqrt(sum(blueDistance_r .^ 2, 2));
            blueDistance_r = min(blueDistance_r);
            blueCentrality_r = log(1 + (1/blueDistance_r));
            
        else
            
            blueWidth_r = 0;
            blueArea_r = 0;
            bluePercentage_r = 0;
            blueCentrality_r = 0;
            
        end
        
        % Update measurements:
        blueMeasurements(r, 1) = bluePresence_r;
        blueMeasurements(r, 2) = blueWidth_r;
        blueMeasurements(r, 3) = blueArea_r;
        blueMeasurements(r, 4) = bluePercentage_r;
        blueMeasurements(r, 5) = blueCentrality_r;
        
    end
    
    % Update dataset:
    measurements = [whiteMeasurements; blueMeasurements];
    data(i).MEASUREMENTS = measurements;
    
end

%%%%%%%%%%%%%%%%%%%%%% REMOVE NAN MEASUREMENTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialise:
removeIdxs = [];


% Check if measurements are NaN:
for i = 1:nData
    measurements = data(i).MEASUREMENTS;
    if isnan(sum(measurements(:)))
        removeIdxs = [removeIdxs, i];
    end
end

% Update dataset:
data(removeIdxs) = [];
nData = length(data);

% Save:
save(pathToDataset,'-v7.3','data');
disp(['Dataset updated with measurements and ',num2str(length(removeIdxs)),' removed.']);
disp(['Dataset has ',num2str(nData),' pairs of images, BCVA, and measurements.']);

%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE STATISTICS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialise:
accumMeasurements = zeros([nWhiteROIs + nBlueROIs, 5, nData]);

% Accumulate:
for i = 1:nData
    accumMeasurements(:,:,i) = data(i).MEASUREMENTS;
end

% Calculate mean and standard deviation:
mean_vector = mean(accumMeasurements, 3);
std_vector = std(accumMeasurements, [], 3);

% Set mean = 0 and std = 1 for binary variables:
mean_vector(:, 1) = 0;
std_vector(:, 1) = 1;

% Save:
[baseFolder, baseName, ~] = fileparts(pathToDataset);
saveFilename = fullfile(baseFolder, [baseName,'_stats.mat']);
save(saveFilename,'-v7.3','mean_vector','std_vector');
disp('Dataset statistics saved.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [widths, area, centers] = getMeasurements(mask)

% Area:
area = sum(mask(:));

% Width, centers:
widths = [];
centers = [];

cc = regionprops(mask);
for r = 1:length(cc)
    
    box = cc(r).BoundingBox;    % [x, y, w, h]
    widths = [widths, box(3)];
    
    ctr = cc(r).Centroid;       % [x, y]
    centers = [centers; ctr];
end
end
