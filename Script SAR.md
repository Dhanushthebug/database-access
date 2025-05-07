############################################################################################################3

Sure! Below is a **complete and final MATLAB script** that:

* Loads your grayscale satellite image
* Performs **K-means clustering** for 5 land cover types
* Uses **interactive ROI input** to map clusters to semantic classes:
  **Background**, **Urban**, **Vegetation**, **Water**, **Roads**
* Generates a **ground truth mask**
* Overlays it for visual validation
* Optionally saves individual binary masks (for each class)

---

### ✅ **Final Script: `generate_ground_truth_mask.m`**

```matlab
% ---- Step 1: Load and Preprocess Image ----
img = imread('download_bw.png');

if size(img, 3) == 3
    grayImg = rgb2gray(img);
else
    grayImg = img;
end

grayImg = imadjust(grayImg); % Enhance contrast

% ---- Step 2: Perform K-means Clustering ----
imgDouble = double(grayImg(:));
nClasses = 5;

[clusterIdx, clusterCenters] = kmeans(imgDouble, nClasses, ...
    'MaxIter', 500, 'Replicates', 3);

segmented = reshape(clusterIdx, size(grayImg));

% ---- Step 3: Use ROI to Map Clusters to Semantic Classes ----
classNames = {'Background', 'Urban', 'Vegetation', 'Water', 'Roads'};
classMap = mapClustersToClasses(grayImg, segmented, classNames);

% ---- Step 4: Generate Ground Truth Mask ----
groundTruth = zeros(size(segmented));
for i = 1:numel(classMap)
    if classMap(i) ~= 0
        groundTruth(segmented == i) = classMap(i);
    end
end

% ---- Step 5: Overlay for Visual Validation ----
cmap = [
    0   0   0;      % 1 - Background (black)
    1   0   0;      % 2 - Urban (red)
    0   1   0;      % 3 - Vegetation (green)
    0   0   1;      % 4 - Water (blue)
    1   1   0       % 5 - Roads (yellow)
];

grayRGB = repmat(mat2gray(grayImg), [1 1 3]);
labelRGB = label2rgb(groundTruth, cmap, 'k');
alpha = 0.5;
overlayedImage = (1 - alpha) * grayRGB + alpha * im2double(labelRGB);

figure;
imshow(overlayedImage);
title('Overlay: Ground Truth Mask on Grayscale Image');

% ---- Step 6: Save Final Mask ----
imwrite(uint8(groundTruth), 'ground_truth_mask.png');

% ---- Step 7 (Optional): Save Binary Masks Per Class ----
for i = 1:5
    classMask = groundTruth == i;
    imwrite(classMask, sprintf('class_%d_mask.png', i));  % or use classNames{i}
end
```

---

### ✅ **Also Save This Function Separately: `mapClustersToClasses.m`**

```matlab
function classMap = mapClustersToClasses(grayImg, segmented, classNames)
    nClasses = length(classNames);
    classMap = zeros(1, nClasses);
    usedClusters = [];

    figure;
    imshow(grayImg);
    title('Draw ROI for each class when prompted');

    for i = 1:nClasses
        prompt = ['Draw ROI for "', classNames{i}, '" and double-click to finish...'];
        disp(prompt);
        roi = drawpolygon('DisplayName', classNames{i});
        mask = createMask(roi);

        roiClusters = segmented(mask);
        clusterLabel = mode(roiClusters);

        if ismember(clusterLabel, usedClusters)
            warning(['Cluster label ', num2str(clusterLabel), ' already assigned. Skipping ', classNames{i}]);
        else
            classMap(clusterLabel) = i;
            usedClusters(end+1) = clusterLabel;
        end
    end

    close;
    disp('Cluster-to-Class Mapping:');
    disp(classMap);
end
```

---

###################################################################################################################

Great — here's the updated **batch processing script** that:

* Works with `.png`, `.jpg`, and `.tif` images
* Saves each **ground truth mask as both `.jpg` and `.mat`**
* Optionally exports **individual class binary masks** as `.jpg`

---

### ✅ Final Script: `batch_generate_ground_truth_masks.m`

```matlab
% ---- Settings ----
inputFolder = 'images';       % Folder containing input images
outputFolder = 'masks';       % Output folder for masks
mkdir(outputFolder);

% Support multiple file types
filePattern = [dir(fullfile(inputFolder, '*.png'));
               dir(fullfile(inputFolder, '*.jpg'));
               dir(fullfile(inputFolder, '*.tif'))];

% ---- Initialize ----
reuseClassMap = false;
classNames = {'Background', 'Urban', 'Vegetation', 'Water', 'Roads'};

for k = 1:length(filePattern)
    filename = filePattern(k).name;
    filepath = fullfile(inputFolder, filename);
    [~, nameOnly, ~] = fileparts(filename);
    disp(['Processing: ', filename]);

    % Step 1: Load and preprocess
    img = imread(filepath);
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    grayImg = imadjust(grayImg);

    % Step 2: K-means clustering
    imgDouble = double(grayImg(:));
    [clusterIdx, ~] = kmeans(imgDouble, 5, 'MaxIter', 500, 'Replicates', 3);
    segmented = reshape(clusterIdx, size(grayImg));

    % Step 3: Draw ROIs only once on the first image
    if ~reuseClassMap
        classMap = mapClustersToClasses(grayImg, segmented, classNames);
        reuseClassMap = true;
    end

    % Step 4: Generate ground truth mask
    groundTruth = zeros(size(segmented));
    for i = 1:numel(classMap)
        if classMap(i) ~= 0
            groundTruth(segmented == i) = classMap(i);
        end
    end

    % Step 5: Save ground truth mask
    mask_jpg_path = fullfile(outputFolder, [nameOnly, '_mask.jpg']);
    imwrite(uint8(groundTruth), mask_jpg_path);

    mask_mat_path = fullfile(outputFolder, [nameOnly, '_mask.mat']);
    save(mask_mat_path, 'groundTruth');

    % Step 6 (Optional): Save class-specific masks as .jpg
    for i = 1:5
        classMask = uint8(groundTruth == i) * 255; % binary to image format
        imwrite(classMask, fullfile(outputFolder, ...
            sprintf('%s_class_%d.jpg', nameOnly, i)));
    end
end
```

---

### ✅ Recap of Outputs (per image)

* **`image_mask.jpg`** → class-labeled ground truth (values 1–5 as gray levels)
* **`image_mask.mat`** → same mask stored as a MATLAB variable (`groundTruth`)
* **`image_class_i.jpg`** → binary masks per class (optional)

---

