function ImageComparisonGUI
    % Creating the main GUI window
    fig = uifigure('Name', 'Image Comparison GUI', 'Position', [100 100 1200 700]);

    % Dropdown for selecting binarization method
    lblMethod = uilabel(fig, 'Position', [50 650 150 30], 'Text', 'Select Method:');
    methodDropdown = uidropdown(fig, 'Position', [150 650 200 30], 'Items', {
        'Otsu', 'Sauvola', 'Mean Thresholding', 'Entropy-Based', 'Niblack', 'Bernsen', 'Adaptive Gaussian'}, 'Value', 'Otsu');

    % Button for uploading image
    btnLoadImage = uibutton(fig, 'Position', [50 600 150 30], 'Text', 'Load Image', 'ButtonPushedFcn', @(~,~) loadImage());

    % Button for uploading GroundTruth
    btnLoadGT = uibutton(fig, 'Position', [250 600 150 30], 'Text', 'Load GroundTruth', 'ButtonPushedFcn', @(~,~) loadGroundTruth());

    % Button for performing analysis
    btnAnalyze = uibutton(fig, 'Position', [450 600 150 30], 'Text', 'Analyze', 'ButtonPushedFcn', @(~,~) analyze());

    % Panels for displaying images
    axOriginal = uiaxes(fig, 'Position', [50 300 300 300]);
    title(axOriginal, 'Original Image');
    axOriginal.XAxis.Visible = 'off';
    axOriginal.YAxis.Visible = 'off';

    axGroundTruth = uiaxes(fig, 'Position', [400 300 300 300]);
    title(axGroundTruth, 'Ground Truth');
    axGroundTruth.XAxis.Visible = 'off';
    axGroundTruth.YAxis.Visible = 'off';

    axProcessed = uiaxes(fig, 'Position', [750 300 300 300]);
    title(axProcessed, 'Processed Image');
    axProcessed.XAxis.Visible = 'off';
    axProcessed.YAxis.Visible = 'off';

     % Panel for displaying metrics
    metricsPanel = uipanel(fig, 'Position', [1050 250 150 400], 'Title', 'Metrics', 'FontSize', 12);
    lblPrecision = uilabel(metricsPanel, 'Position', [10 360 120 20], 'Text', 'Precision: 0.00');
    lblRecall = uilabel(metricsPanel, 'Position', [10 320 120 20], 'Text', 'Recall: 0.00');
    lblFMeasure = uilabel(metricsPanel, 'Position', [10 280 120 20], 'Text', 'F-Measure: 0.00');
    lblAccuracy = uilabel(metricsPanel, 'Position', [10 240 120 20], 'Text', 'Accuracy: 0.00');
    lblSpecificity = uilabel(metricsPanel, 'Position', [10 200 120 20], 'Text', 'Specificity: 0.00');
    lblIoU = uilabel(metricsPanel, 'Position', [10 160 120 20], 'Text', 'IoU: 0.00');
    lblDice = uilabel(metricsPanel, 'Position', [10 120 120 20], 'Text', 'Dice: 0.00');
    lblBalancedAccuracy = uilabel(metricsPanel, 'Position', [10 80 120 20], 'Text', 'Balanced Acc: 0.00');
    lblMCC = uilabel(metricsPanel, 'Position', [10 40 120 20], 'Text', 'MCC: 0.00');
    lblFNR = uilabel(metricsPanel, 'Position', [10 0 120 20], 'Text', 'FNR: 0.00');

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
        if size(groundTruth, 3) == 3
            groundTruth = imbinarize(rgb2gray(groundTruth));
        elseif ~islogical(groundTruth)
            groundTruth = imbinarize(groundTruth);
        end
        if ~isempty(originalImage) && ~isequal(size(groundTruth), size(originalImage))
            uialert(fig, 'The sizes of the original image and the ground truth image do not match. Please select an image with matching dimensions.', 'Size Mismatch');
            groundTruth = []; % Reset ground truth if size mismatch
            return;
        end
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

            case 'Niblack'
                T = adaptthresh(originalImage, 0.5, 'ForegroundPolarity', 'dark', 'Statistic', 'mean');
                binaryImage = imbinarize(originalImage, T);

            case 'Bernsen'
                % Define the local neighborhood size
                windowSize = 31; % Adjust this as needed
                % Compute local mean and contrast
                localMean = imfilter(double(originalImage), ones(windowSize) / (windowSize ^ 2), 'symmetric');
                localMax = ordfilt2(originalImage, windowSize ^ 2, ones(windowSize));
                localMin = ordfilt2(originalImage, 1, ones(windowSize));
                localContrast = (localMax - localMin) / 2;
                % Set a threshold for local contrast
                contrastThreshold = 15; % Adjust as needed
                binaryImage = (originalImage > localMean) & (localContrast > contrastThreshold);      
            

            case 'Adaptive Gaussian'
                T = adaptthresh(originalImage, 0.5, 'ForegroundPolarity', 'dark', 'NeighborhoodSize', [25 25]);
                binaryImage = imbinarize(originalImage, T);

            otherwise
                uialert(fig, 'Unsupported method selected.', 'Error');
                return;
        end

        % Display the processed image
        imshow(binaryImage, 'Parent', axProcessed);

        % Compute and display metrics
        [precision, recall, fMeasure, accuracy, specificity, IoU, dice, balancedAccuracy, MCC, FNR] = computeMetrics(binaryImage, groundTruth);
        lblPrecision.Text = sprintf('Precision: %.2f', precision);
        lblRecall.Text = sprintf('Recall: %.2f', recall);
        lblFMeasure.Text = sprintf('F-Measure: %.2f', fMeasure);
        lblAccuracy.Text = sprintf('Accuracy: %.2f', accuracy);
        lblSpecificity.Text = sprintf('Specificity: %.2f', specificity);
        lblIoU.Text = sprintf('IoU: %.2f', IoU);
        lblDice.Text = sprintf('Dice: %.2f', dice);
        lblBalancedAccuracy.Text = sprintf('Balanced Acc: %.2f', balancedAccuracy);
        lblMCC.Text = sprintf('MCC: %.2f', MCC);
        lblFNR.Text = sprintf('FNR: %.2f', FNR);

    end

   % --- Metrics calculation function ---
    function [precision, recall, fMeasure, accuracy, specificity, IoU, dice, balancedAccuracy, MCC, FNR] = computeMetrics(binaryImg, gt)
        if ~isequal(size(binaryImg), size(gt))
            error('Binary image and ground truth sizes do not match.');
        end

        TP = sum((binaryImg == 1) & (gt == 1), 'all');
        FP = sum((binaryImg == 1) & (gt == 0), 'all');
        FN = sum((binaryImg == 0) & (gt == 1), 'all');
        TN = sum((binaryImg == 0) & (gt == 0), 'all');

        % Standard metrics
        precision = TP / max(TP + FP, 1);
        recall = TP / max(TP + FN, 1);
        accuracy = (TP + TN) / max(TP + FP + FN + TN, 1);
        specificity = TN / max(TN + FP, 1);
        IoU = TP / max(TP + FP + FN, 1);

        % Additional metrics
        if precision + recall == 0
            fMeasure = 0;
        else
            fMeasure = 2 * (precision * recall) / (precision + recall);
        end

        dice = (2 * TP) / max(2 * TP + FP + FN, 1); % Dice Coefficient
        balancedAccuracy = (recall + specificity) / 2; % Balanced Accuracy

        MCC = (TP * TN - FP * FN) / sqrt(max((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), 1)); % Matthews Correlation Coefficient
        FNR = FN / max(FN + TP, 1); % False Negative Rate
    end
end
