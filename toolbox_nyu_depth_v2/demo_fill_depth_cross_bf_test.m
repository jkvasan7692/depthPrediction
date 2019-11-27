% Demo's the in-painting function fill_depth_cross_bf.m

DATASET_PATH = '/media/kirthi/Seagate Backup Plus Drive/MLProjectDataset/nyu_depth_v2_labeled.mat';

load(DATASET_PATH, 'images', 'rawDepths', 'depths');

%%
imageInd = 1;

imgRgb = images(:,:,:,imageInd);
imgDepthAbs = rawDepths(:,:,imageInd);
imgDepthsProcessed = depths(:,:,imageInd);

% Crop the images to include the areas where we have depth information.
% imgRgb = crop_image(imgRgb);
% imgDepthAbs = crop_image(imgDepthAbs);

imgDepthFilled = fill_depth_cross_bf(imgRgb, double(imgDepthAbs));

figure(1);
subplot(1,4,1); imagesc(imgRgb);
subplot(1,4,2); imagesc(imgDepthAbs);
subplot(1,4,3); imagesc(imgDepthsProcessed);
subplot(1,4,4); imagesc(double(imgDepthFilled));