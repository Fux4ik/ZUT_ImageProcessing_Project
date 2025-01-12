function ImageComparisonGUI
    % Creating the main GUI window
    fig = uifigure('Name', 'Image Comparison GUI', 'Position', [100 100 1000 600]);

    % Dropdown for selecting binarization method
    lblMethod = uilabel(fig, 'Position', [50 550 150 30], 'Text', 'Select Method:');
    methodDropdown = uidropdown(fig, 'Position', [150 550 150 30], ...
        'Items', {'Otsu', 'Sauvola', 'Mean Thresholding', 'Entropy-Based'}, 'Value', 'Otsu');

    % Button for uploading image
    btnLoadImage = uibutton(fig, 'Position', [50 500 150 30], ...
        'Text', 'Load Image', 'ButtonPushedFcn', @(~,~) loadImage());

    % Button for uploading GroundTruth
    btnLoadGT = uibutton(fig, 'Position', [250 500 150 30], ...
        'Text', 'Load GroundTruth', 'ButtonPushedFcn', @(~,~) loadGroundTruth());

    % Button for performing analysis
    btnAnalyze = uibutton(fig, 'Position', [450 500 150 30], ...
        'Text', 'Analyze', 'ButtonPushedFcn', @(~,~) analyze());

    % Panels for displaying images
    axOriginal = uiaxes(fig, 'Position', [50 250 200 200]);
    title(axOriginal, 'Original Image');
    axGroundTruth = uiaxes(fig, 'Position', [300 250 200 200]);
    title(axGroundTruth, 'Ground Truth');
    axProcessed = uiaxes(fig, 'Position', [550 250 200 200]);
    title(axProcessed, 'Processed Image');

    % Panel for displaying metrics
    metricsPanel = uipanel(fig, 'Position', [800 200 180 250], 'Title', 'Metrics', 'FontSize', 12);

    % Labels for individual metrics
    lblPrecision = uilabel(metricsPanel, 'Position', [10 200 160 20], 'Text', 'Precision: 0.00');
    lblRecall = uilabel(metricsPanel, 'Position', [10 170 160 20], 'Text', 'Recall: 0.00');
    lblFMeasure = uilabel(metricsPanel, 'Position', [10 140 160 20], 'Text', 'F-Measure: 0.00');
    lblAccuracy = uilabel(metricsPanel, 'Position', [10 110 160 20], 'Text', 'Accuracy: 0.00');
    lblSpecificity = uilabel(metricsPanel, 'Position', [10 80 160 20], 'Text', 'Specificity: 0.00');
    lblIoU = uilabel(metricsPanel, 'Position', [10 50 160 20], 'Text', 'IoU: 0.00');

    % Global variables for saving data
    originalImage = [];
    groundTruth = [];

    % --- Function for loading image ---
    function loadImage()
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tiff;*.tif', 'Image Files (*.jpg, *.png, *.bmp, *.tiff, *.tif)'}, 'Select an Image');
        if isequal(file, 0)
            return;
        end
        originalImage = imread(fullfile(path, file));
        if size(originalImage, 3) == 3
            originalImage = rgb2gray(originalImage);
        end
        imshow(originalImage, 'Parent', axOriginal);
    end

    % --- Function for loading GroundTruth ---
    function loadGroundTruth()
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tiff;*.tif', 'Ground Truth Files (*.jpg, *.png, *.bmp, *.tiff, *.tif)'}, 'Select Ground Truth');
        if isequal(file, 0)
            return;
        end
        groundTruth = imread(fullfile(path, file));

        % Convert to binary image if not a boolean type
        if size(groundTruth, 3) == 3
            groundTruth = imbinarize(rgb2gray(groundTruth));
        elseif ~islogical(groundTruth)
            groundTruth = imbinarize(groundTruth);
        end

        % Resize groundTruth to match originalImage size
        if ~isempty(originalImage) && ~isequal(size(groundTruth), size(originalImage))
            groundTruth = imresize(groundTruth, size(originalImage));
        end

        % Display Ground Truth
        imshow(groundTruth, 'Parent', axGroundTruth);
    end

    % --- Function for performing analysis ---
    function analyze()
        if isempty(originalImage) || isempty(groundTruth)
            uialert(fig, 'Please load both the image and the ground truth.', 'Missing Data');
            return;
        end

        % Get selected method
        selectedMethod = methodDropdown.Value;

        % Perform binarization based on the selected method
        switch selectedMethod
            case 'Otsu'
                level = graythresh(originalImage);
                binaryImage = imbinarize(originalImage, level);

            case 'Sauvola'
                T = adaptthresh(originalImage, 0.5, 'ForegroundPolarity', 'dark', 'Statistic', 'gaussian');
                binaryImage = imbinarize(originalImage, T);

            case 'Mean Thresholding'
                meanThresh = mean2(originalImage) / 255;
                binaryImage = imbinarize(originalImage, meanThresh);

            case 'Entropy-Based'
                entropyFilter = entropyfilt(originalImage);
                threshold = graythresh(entropyFilter);
                binaryImage = imbinarize(entropyFilter, threshold);

            otherwise
                uialert(fig, 'Unsupported method selected.', 'Error');
                return;
        end

        % Display the processed image
        imshow(binaryImage, 'Parent', axProcessed);

        % Check dimensions before calculating metrics
        if ~isequal(size(binaryImage), size(groundTruth))
            uialert(fig, 'Image and ground truth sizes do not match.', 'Size Mismatch');
            return;
        end

        % Compute and display metrics
        [precision, recall, fMeasure, accuracy, specificity, IoU] = computeMetrics(binaryImage, groundTruth);
        lblPrecision.Text = sprintf('Precision: %.2f', precision);
        lblRecall.Text = sprintf('Recall: %.2f', recall);
        lblFMeasure.Text = sprintf('F-Measure: %.2f', fMeasure);
        lblAccuracy.Text = sprintf('Accuracy: %.2f', accuracy);
        lblSpecificity.Text = sprintf('Specificity: %.2f', specificity);
        lblIoU.Text = sprintf('IoU: %.2f', IoU);
    end

    % --- Metrics calculation function ---
    function [precision, recall, fMeasure, accuracy, specificity, IoU] = computeMetrics(binaryImg, gt)
        if ~isequal(size(binaryImg), size(gt))
            error('Binary image and ground truth sizes do not match.');
        end

        TP = sum((binaryImg == 1) & (gt == 1), 'all'); % True Positives
        FP = sum((binaryImg == 1) & (gt == 0), 'all'); % False Positives
        FN = sum((binaryImg == 0) & (gt == 1), 'all'); % False Negatives
        TN = sum((binaryImg == 0) & (gt == 0), 'all'); % True Negatives

        % Debugging output for diagnostics
        fprintf('TP: %d, FP: %d, FN: %d, TN: %d\n', TP, FP, FN, TN);

        % Avoid division by zero
        precision = TP / max(TP + FP, 1);
        recall = TP / max(TP + FN, 1);
        accuracy = (TP + TN) / max(TP + FP + FN + TN, 1);
        specificity = TN / max(TN + FP, 1);
        IoU = TP / max(TP + FP + FN, 1);

        if precision + recall == 0
            fMeasure = 0;
        else
            fMeasure = 2 * (precision * recall) / (precision + recall);
        end
    end
end
