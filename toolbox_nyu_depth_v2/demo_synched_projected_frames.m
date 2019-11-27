% The directory where you extracted the raw dataset.
datasetDir = '/media/kirthi/Seagate Backup Plus Drive/MLProjectDataset';

sceneDirs = GetSubDirsFirstLevelOnly(datasetDir);

structInd = 1;

imgRgbToStruct = zeros(480, 640, 3, 5000, 'uint8');
imgRawDepthToStruct = zeros(480, 640,5000, 'single');
imgDepthsToStruct = zeros(480, 640,5000, 'single');

%%
for dirInd = 6:10
    % The name of the scene to demo.
    sceneName = 'cafe_0001a';

    % The absolute directory of the 
    sceneDir = sprintf('%s/%s', datasetDir, sceneDirs{dirInd})

% Reads the list of frames.
    frameList = get_synched_frames(sceneDir);
    rawRgbFileName = {}
    rawDepthFileName = {}
%%
% Displays each pair of synchronized RGB and Depth frames.
    tic;
    for ii = 1 : 2 : numel(frameList)
        
      imgRgb = imread([sceneDir '/' frameList(ii).rawRgbFilename]);
      imgDepthRaw = swapbytes(imread([sceneDir '/' frameList(ii).rawDepthFilename]));
      imgDepthProj = project_depth_map(imgDepthRaw, imgRgb);
      % Crop the images to include the areas where we have depth information.
%       imgRgb = crop_image(imgRgb);  
      imgDepthAbs = crop_image(imgDepthProj);

      imgDepthFilled = fill_depth_colorization(imgRgb, (imgDepthProj));
      
      imgRgbToStruct(:, :, :, structInd) = imgRgb;
      imgRawDepthToStruct(:, :, structInd) = imgDepthProj;
      imgDepthsToStruct(:, :, structInd) = imgDepthFilled;
      rawRgbFileName{structInd,1} = sceneDirs{dirInd} + "/" + frameList(ii).rawRgbFilename;
      rawDepthFileName{structInd,1} = sceneDirs{dirInd} + "/" + frameList(ii).rawDepthFilename;
      
      structInd = structInd+1;

%       figure(1);
%       % Show the RGB image.
%       subplot(1,3,1);
%       imagesc(imgRgb);
%       axis off;
%       axis equal;
%       title('RGB');
% 
%       % Show the Raw Depth image.
%       subplot(1,3,2);
%       imagesc(imgDepthRaw);
%       axis off;
%       axis equal;
%       title('Raw Depth');
%       caxis([800 1100]);
% 
%       % Show the projected depth image.
%       imgDepthProj = project_depth_map(imgDepthRaw, imgRgb);
%       subplot(1,3,3);
%       imagesc(imgDepthFilled);
%       axis off;
%       axis equal;
%       title('Projected Depth');
% 
%       pause(0.01);
    end
      toc
      structInd
end
depths = imgDepthsToStruct(:,:,1:structInd-1);
rawDepth = imgRawDepthToStruct(:,:,1:structInd-1);
images = imgRgbToStruct(:,:,:,1:structInd-1);

save('myfile_2.mat', 'images', 'depths', 'rawDepth', 'rawRgbFileName', 'rawDepthFileName');

function [subDirsNames] = GetSubDirsFirstLevelOnly(parentDir)
    % Get a list of all files and folders in this folder.
    files = dir(parentDir);
    % Get a logical vector that tells which is a directory.
    dirFlags = [files.isdir];
    % Extract only those that are directories.
    subDirs = files(dirFlags);
    subDirsNames = cell(1, numel(subDirs) - 2);
    for i=3:numel(subDirs)
        subDirsNames{i-2} = subDirs(i).name;
    end
end
