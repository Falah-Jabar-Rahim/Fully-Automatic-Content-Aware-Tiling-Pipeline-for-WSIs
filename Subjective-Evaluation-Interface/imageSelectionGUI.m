
function imageSelectionGUI(folderPath)
    % List all image files in the folder
    imageFiles = dir(fullfile(folderPath, '*.bmp')); % or '*.png' or other formats
    imageSets = loadImageSets(imageFiles);
    numSets = numel(imageSets);
    bestImages = cell(numSets, 1);
    
    % Create the UI figure
    fig = uifigure('Name', 'Image Quality Selector');
    fig.Position = [100 100 1200 600];
    
    % Create a grid layout
    gl = uigridlayout(fig, [3, 2], 'RowHeight', {'1x', '1x', 40});
    
    % Axes for displaying images
    ax1 = uiaxes(gl);
    ax2 = uiaxes(gl);
    ax1.Layout.Row = 1;
    ax1.Layout.Column = 1;
    ax2.Layout.Row = 1;
    ax2.Layout.Column = 2;
    
    % Label for instructions
    lbl = uilabel(gl, 'Text', 'Select the image with the best quality:', 'FontSize', 16);
    lbl.Layout.Row = 2;
    lbl.Layout.Column = [1, 2];
    
    % Next button
    nextBtn = uibutton(gl, 'Text', 'Next Set', 'ButtonPushedFcn', @(btn,event) nextSet());
    nextBtn.Layout.Row = 3;
    nextBtn.Layout.Column = 1;
    
    % Done button
    doneBtn = uibutton(gl, 'Text', 'Done', 'ButtonPushedFcn', @(btn,event) doneSelection());
    doneBtn.Layout.Row = 3;
    doneBtn.Layout.Column = 2;
    
    % Variables to keep track of progress
    currentSet = 1;
    currentPair = 1;
    pairs = [1 2; 1 3; 1 4; 1 5; 2 3; 2 4; 2 5; 3 4; 3 5; 4 5];
    numPairs = size(pairs, 1);
    selectedImages = zeros(numPairs, 1);
    
    % Show the first pair
    showPair();
    
    function showPair()
        % Get the images to compare
        setImages = imageSets{currentSet};
        pair = pairs(currentPair, :);
        img1 = setImages{pair(1)};
        img2 = setImages{pair(2)};
        
        % Display images
        imshow(img1, 'Parent', ax1);
        imshow(img2, 'Parent', ax2);
        
        % Ask user to select the best image
        choice = uiconfirm(fig, 'Select the best image', 'Choose Image', ...
            'Options', {'Left', 'Right'}, ...
            'DefaultOption', 1, 'CancelOption', 2);
        
        if strcmp(choice, 'Left')
            selectedImages(currentPair) = pair(1);
        else
            selectedImages(currentPair) = pair(2);
        end
        
        % Move to the next pair or set
        if currentPair < numPairs
            currentPair = currentPair + 1;
            showPair();
        else
            % Determine the best image from selected pairs
            bestImage = mode(selectedImages);
            bestImages{currentSet} = setImages{bestImage};
            
            currentPair = 1;
            if currentSet < numSets
                currentSet = currentSet + 1;
                showPair();
            else
                uialert(fig, 'All sets completed!', 'Done');
            end
        end
    end

    function nextSet()
        if currentSet <= numSets
            showPair();
        end
    end

    function doneSelection()
        disp('Best images selected for each set:');
        for i = 1:numSets
            disp(['Set ', num2str(i), ':']);
            imshow(bestImages{i});
            pause(1);
        end
        close(fig);
    end
function imageSets = loadImageSets(imageFiles)
    % Initialize an empty cell array for image sets
    imageSets = {};

    % Regular expression to match filenames and extract the set number
    pattern = 'set(\d+)_(center|left|right|top|bottom)\.jpg';

    % Create a temporary structure to hold images by set number
    tempSets = struct();

    % Loop through all image files
    for i = 1:length(imageFiles)
        % Match the filename to the pattern
        tokens = regexp(imageFiles(i).name, pattern, 'tokens');

        if ~isempty(tokens)
            % Extract set number and position
            setNumber = str2double(tokens{1}{1});
            position = tokens{1}{2};

            % Load the image
            img = imread(fullfile(imageFiles(i).folder, imageFiles(i).name));

            % Store the image in the temporary structure
            if ~isfield(tempSets, ['set', num2str(setNumber)])
                tempSets.(['set', num2str(setNumber)]) = struct();
            end
            tempSets.(['set', num2str(setNumber)]).(position) = img;
        else
            % Display a warning if the file does not match the expected pattern
            disp(['File "' imageFiles(i).name '" does not match the expected pattern and will be skipped.']);
        end
    end

    % Convert the structure into a cell array of image sets
    setNumbers = fieldnames(tempSets);
    for i = 1:length(setNumbers)
        currentSet = tempSets.(setNumbers{i});
        if isfield(currentSet, 'center') && isfield(currentSet, 'left') && ...
           isfield(currentSet, 'right') && isfield(currentSet, 'top') && ...
           isfield(currentSet, 'bottom')
            imageSets{i} = {currentSet.center, currentSet.left, currentSet.right, currentSet.top, currentSet.bottom};
        else
            disp(['Set "' setNumbers{i} '" is missing one or more images and will be skipped.']);
        end
    end
end


end
