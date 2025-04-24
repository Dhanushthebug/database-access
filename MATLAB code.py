%% Land Cover Classification from SAR Image using U-Net
clc; clear;

%% Step 1: Load Image and Generate Initial Mask
img = imread('Busan_KV.jpg');
img_gray = rgb2gray(img);
img_double = im2double(img_gray);
[rows, cols] = size(img_gray);

% K-means clustering
pixels = img_double(:);
nClusters = 4;
[idx, C] = kmeans(pixels, nClusters, 'Replicates', 5);

% Reshape and label clusters
clustered = reshape(idx, rows, cols);
[~, sortedIdx] = sort(C);

label_map = zeros(size(clustered));
label_map(clustered == sortedIdx(1)) = 1; % Water
label_map(clustered == sortedIdx(2)) = 2; % Vegetation
label_map(clustered == sortedIdx(3)) = 3; % Barren
label_map(clustered == sortedIdx(4)) = 4; % Urban

% Save mask
imwrite(uint8(label_map), 'Busan_KV_mask.png');
disp('Initial mask saved as Busan_KV_mask.png');

%% Step 2: Launch Image Labeler for Manual Correction
disp('Launch the Image Labeler to refine the mask manually.');
imageLabeler;

% ===== MANUAL STEPS INSIDE APP =====
% 1. Add Image: Busan_KV.jpg
% 2. Import Labels: Busan_KV_mask.png
% 3. Refine the mask (draw/edit)
% 4. Export Pixel Label Data to LabeledData/pixelLabelData
% ===================================

% Pause for user to complete labeling
input('After exporting labels, press ENTER to continue...');

%% Step 3: Load Data for Training
imageFolder = fullfile('LabeledData', 'images');
labelFolder = fullfile('LabeledData', 'pixelLabelData');

classNames = ["Water", "Vegetation", "Barren", "Urban"];
labelIDs = [1, 2, 3, 4];

imds = imageDatastore(imageFolder);
pxds = pixelLabelDatastore(labelFolder, classNames, labelIDs);
trainingData = pixelLabelImageDatastore(imds, pxds);

%% Step 4: Resize (Optional) and Train U-Net
imageSize = [256 256 1];
augmentedTrainingData = transform(trainingData, @(data) preprocess(data, imageSize));

lgraph = unetLayers(imageSize, numel(classNames));

options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',30, ...
    'MiniBatchSize',4, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'VerboseFrequency',10);

net = trainNetwork(augmentedTrainingData, lgraph, options);

%% Step 5: Predict and Visualize
testImg = imread('Busan_KV.jpg');
testImg = imresize(rgb2gray(testImg), imageSize(1:2));
predictedMask = semanticseg(testImg, net);

cmap = [0 0 1; 0 1 0; 1 1 0; 1 0 0]; % BGRY for Water, Vegetation, Barren, Urban
overlay = labeloverlay(testImg, predictedMask, 'Colormap', cmap, 'Transparency', 0.4);
imshow(overlay);
title('Predicted Land Cover Segmentation');

%% Helper Function
function dataOut = preprocess(data, targetSize)
    dataOut = data;
    dataOut.InputImage = imresize(data.InputImage, targetSize(1:2));
    dataOut.PixelLabelImage = imresize(data.PixelLabelImage, targetSize(1:2), 'nearest');
end
