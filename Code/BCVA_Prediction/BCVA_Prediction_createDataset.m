%
% SLIT-Net v2
% DOI: 10.1167/tvst.10.12.2
%
% Create BCVA prediction dataset by matching white-blue pairs with BCVA
%

clear;
close all;
clc;

% User-input:
mainDatasetFolder = '../../Datasets';

% These did not have a manual segmentation of the limbus for comparison,
% and were excluded from analysis even if they have BCVA:
whiteVXXXToExclude = {'V153','V155','V160','V167','V169','V195'};
blueVXXXToExclude = {'V069','V086','V123'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Specific dataset folders:
bcvaDatasetFolder = fullfile(mainDatasetFolder,'BCVA_Prediction');
segmentationDatasetFolder = fullfile(mainDatasetFolder,'Segmentation');

% Load cross reference list:
crossReferenceListFilename = fullfile(bcvaDatasetFolder,'crossReferenceList.mat');
load(crossReferenceListFilename);
[whiteRefList, blueRefList] = buildSimpleRefList(crossReferenceList);

% Initialise:
data = [];
count = 0;

for wk = 1:6
    
    % White dataset filename:
    pathToWhiteDataset = fullfile(segmentationDatasetFolder,'White_Light',['test_data_K',num2str(wk),'.mat']);
    
    % Load white data:
    whiteData = load(pathToWhiteDataset);
    whiteData = whiteData.data;
    nWhiteData = length(whiteData);
    
    for wi = 1:nWhiteData
        
        % Check if it has BCVA:
        if isnan(whiteData(wi).LOGMAR)
            continue;
        end
        
        % Get whiteVXXX:
        [~, whiteVXXX, ~] = fileparts(whiteData(wi).PATH);
        
        % Get matching blueVXXX:
        idx = strcmp(whiteRefList, whiteVXXX);
        if sum(idx(:)) == 0
            continue;
        else
            matchBlueVXXX = blueRefList{idx};
        end
        
        % Check if either should be excluded:
        if ~isempty(find(strcmp(whiteVXXXToExclude,whiteVXXX),1))
           continue; 
        end
        if ~isempty(find(strcmp(blueVXXXToExclude,matchBlueVXXX),1))
            continue;
        end
        
        % Initialise flag:
        matchFound = false;
        
        for bk = 1:6
            
            if matchFound
                break;
            end
            
            % Blue dataset filename:
            pathToBlueDataset = fullfile(segmentationDatasetFolder,'Blue_Light',['test_data_K',num2str(bk),'.mat']);
            
            % Load blue data:
            blueData = load(pathToBlueDataset);
            blueData = blueData.data;
            nBlueData = length(blueData);

            for bi = 1:nBlueData
                
                if matchFound
                    break;
                end
                
                % Get blueVXXX:
                [~, blueVXXX, ~] = fileparts(blueData(bi).PATH);
                
                % Check if match:
                if ~strcmp(blueVXXX, matchBlueVXXX)
                    continue;
                end
                
                % Check if BCVA matches:
                whitelogMAR = whiteData(wi).LOGMAR;
                bluelogMAR = blueData(bi).LOGMAR;
                if ~(whitelogMAR == bluelogMAR)
                    disp(['ERROR: White ',whiteVXXX,' and blue ',blueVXXX,' logMAR do not match.']);
                    continue;
                else
                    logMAR = whitelogMAR;
                end            
                
                % Update data:
                count = count + 1;
                data(count).WHITE_PATH = whiteData(wi).PATH;
                data(count).BLUE_PATH = blueData(bi).PATH;
                data(count).LOGMAR = logMAR;
                
                matchFound = true;
                disp(['MATCH FOUND | White ',whiteVXXX,' - Blue ',blueVXXX]);
                
            end
            
        end
        
    end
    
end

% Save:
saveFilename = fullfile(bcvaDatasetFolder,'measurements.mat');
save(saveFilename,'-v7.3','data');
disp(['Dataset created with ',num2str(count),' pairs of images and BCVA.']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [whiteRefList, blueRefList] = buildSimpleRefList(refList)

nRefs = length(refList);

% Initialise:
whiteRefList = cell([1,nRefs]);
blueRefList = cell([1,nRefs]);

for r = 1:nRefs
    
    whiteVXXX = sprintf('V%03d', refList(r).WHITE_VOLUME);
    blueVXXX = sprintf('V%03d', refList(r).BLUE_VOLUME);
    
    whiteRefList{r} = whiteVXXX;
    blueRefList{r} = blueVXXX;
    
end

end
