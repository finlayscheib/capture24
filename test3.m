%% Data Loading 
clc; clear;
data = imageDatastore("/exports/eddie/scratch/s2190468/capture24/data/GAF/", "IncludeSubfolders",true);
%retrieve file names
fileNames = data.Files;

% creating labels from file names and then attaching to image data
labels = categorical(zeros(length(fileNames),1));
for i = 1:length(fileNames)

    %split into parts and get ending for label
    [~, name, ~]=fileparts(fileNames{i});
    parts = split(name,'_');
    labels(i) = categorical(cellstr(parts{end}));
end
data.Labels = labels;

numFiles = length(fileNames);
participant_num = zeros(numFiles, 1);

for i = 1:numFiles
    [~, name, ~] = fileparts(fileNames{i});
    parts = split(name, '_');
    participant_num(i) = str2double(erase(parts{1}, 'P')); % Extracts "1" from "P001"
end

trainIdx = (participant_num <= 80);
valIdx   = (participant_num > 80 & participant_num <= 100);
testIdx  = (participant_num > 100);

train = subset(data, trainIdx);
val   = subset(data, valIdx);
test  = subset(data, testIdx);

train.Labels = removecats(train.Labels);
val.Labels   = removecats(val.Labels);
test.Labels  = removecats(test.Labels);

% Load resnet101 - Needed to download as eddie can't access online i think
% load('resnet101_initialized.mat', 'net');
% set the parameters
miniBatchSize = 16;
% options = trainingOptions("sgdm", ...
%     "ExecutionEnvironment","gpu", ... %changed from gpu to auto for now 
%     "MiniBatchSize",miniBatchSize, ...
%     "MaxEpochs",50, ... % changed max epochs from 50 to 2
%     "InitialLearnRate",0.0001, ...
%     "Shuffle","every-epoch", ...   
%     "ValidationData",val, ...
%    "ValidationFrequency",30, ... % changed from 30 to 5
%     "Verbose",true, ... % made true for local
%       "Plots","none", ...
%       "Metrics", "accuracy",...
%       "CheckpointPath","/exports/eddie/scratch/s2190468/capture24/outputs/",...
%       "ValidationPatience",10,... %if validation loss goes without improving for 5 checks -> early stopping
%       "OutputNetwork","best-validation"); % choose best val loss network
% % removed: %'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3) but add
% % back in
% 
% disp ('ResNet training started')
% % start training 
% [ResNet_FineTuned, infoResNet] = trainnet(train,net,"crossentropy", options);
load('ResNet_Trained_Model.mat', 'ResNet_FineTuned');
disp('ResNet loaded successfully!');
% % Compute initial performance
% disp('Predicting test set...')
scores = minibatchpredict(ResNet_FineTuned, test); 
classNames = categories(train.Labels);
YPred = scores2label(scores, classNames);
% save('ResNet_Trained_Model.mat', 'ResNet_FineTuned');


% Calculate accuracy
accuracy_CNN = mean(YPred == test.Labels);
disp(['CNN Accuracy: ', num2str(accuracy_CNN)]);

% confusion matrix
% disp('Generating and saving confusion matrix...')
% fig = figure('Visible', 'off'); 
% confusionchart(test.Labels, YPred, Title="CNN Test Set Confusion Matrix");

% save and do not display in eddie
% exportgraphics(fig, 'ResNet_Test_ConfusionMatrix.png', 'Resolution', 300);

% save in case want to plot training progress
% save('resnet_training_log.mat', 'infoResNet');


% RICA

% Features extraction based on Fine-Tuned CNN (net2)
% 
featureLayer =   "pool5";
trainingFeatures_ResNet = minibatchpredict(ResNet_FineTuned,train, Outputs=featureLayer);
validationFeatures_ResNet = minibatchpredict(ResNet_FineTuned,val, Outputs=featureLayer);
testFeatures_ResNet = minibatchpredict(ResNet_FineTuned,test, Outputs=featureLayer);
% 
% if isdlarray(trainingFeatures_ResNet)
%     trainingFeatures_ResNet = extractdata(trainingFeatures_ResNet);
%     validationFeatures_ResNet = extractdata(validationFeatures_ResNet);
%     testFeatures_ResNet = extractdata(testFeatures_ResNet);
% end
% Turns 4D into 2D
trainingFeatures_ResNet = squeeze(trainingFeatures_ResNet);
validationFeatures_ResNet = squeeze(validationFeatures_ResNet);
testFeatures_ResNet = squeeze(testFeatures_ResNet);
% 
% 
 if size(trainingFeatures_ResNet, 1) == 2048
     trainingFeatures_ResNet = trainingFeatures_ResNet';
     validationFeatures_ResNet = validationFeatures_ResNet';
     testFeatures_ResNet = testFeatures_ResNet';
 end
% 
% features input for RICA
disp ('RICA training started')

rng("default") % For reproducibility
q = 100; % number of features obtained from RICA: willactually be 100
Mdl = rica(trainingFeatures_ResNet,q,'IterationLimit',5000); % original iteration limit was 80
trainingFeatures_RICA = transform(Mdl,trainingFeatures_ResNet);
validationFeatures_RICA =  transform(Mdl,validationFeatures_ResNet);
testFeatures_RICA = transform(Mdl,testFeatures_ResNet);


% saving in case crashes
% disp("saving features from RICA")
% save('RICA_Features_checkpoint.mat','trainingFeatures_RICA', 'validationFeatures_RICA', 'testFeatures_RICA', '-v7.3');
% pre-processing for BiLSTM: Each participant into cells

trainingLabels   = train.Labels;
validationLabels = val.Labels;
testLabels       = test.Labels;

participant_num_train = participant_num(participant_num <= 80);
participant_num_val   = participant_num(participant_num > 80 & participant_num <= 100);
participant_num_test  = participant_num(participant_num > 100);

num_train = unique(participant_num_train);
num_val= unique(participant_num_val);
num_test= unique(participant_num_test);

trainingFeatures_RICA_Cell= cell(length(num_train), 1);
trainLabels_Cell= cell(length(num_train), 1);

validationFeatures_RICA_Cell= cell(length(num_val), 1);
validationLabels_Cell= cell(length(num_val), 1);

testFeatures_RICA_Cell= cell(length(num_test), 1);
testLabels_Cell= cell(length(num_test), 1);

for i = 1:length(num_train)
    idx = (participant_num_train == num_train(i)); % Find all rows for this participant
    trainingFeatures_RICA_Cell{i} = trainingFeatures_RICA(idx, :)';
    trainLabels_Cell{i}           = trainingLabels(idx)';
end

for i = 1:length(num_val)
    idx = (participant_num_val == num_val(i));
    validationFeatures_RICA_Cell{i} = validationFeatures_RICA(idx, :)';
    validationLabels_Cell{i}        = validationLabels(idx)';
end

for i = 1:length(num_test)
    idx = (participant_num_test == num_test(i));
    
    testFeatures_RICA_Cell{i} = testFeatures_RICA(idx, :)';
    testLabels_Cell{i} = testLabels(idx)';
end
 
% BiLSTM training
trainLabels=train.Labels;
rng(13);
numFeatures =q; %was 100=q but now changed to 2048 with no RICA
numHiddenUnits = 20; % was 20 - Might increase to 40
numClasses = 4;
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(30, OutputMode="sequence", RecurrentWeightsInitializer="he")
    dropoutLayer(0.2)
    bilstmLayer(15, OutputMode="sequence", RecurrentWeightsInitializer="he")
    dropoutLayer(0.3)
    fullyConnectedLayer(numClasses, WeightsInitializer="he")
    softmaxLayer
];
netLSTM = dlnetwork(layers);

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ... 
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.0001, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.5,...
    'LearnRateDropPeriod',300,...
    'MaxEpochs',2000, ... %was at 1500
    ValidationData={validationFeatures_RICA_Cell,validationLabels_Cell}, ...
    ValidationFrequency=30, ...
    SequenceLength='longest', ...
    Plots='none', ...
    Shuffle='never',...
    InputDataFormats="CTB",...
    GradientThreshold=1,... %hopefully solves the Nan's
    GradientThresholdMethod='absolute-value',...
    TargetDataFormats="CTB",... %removed metrics = accuracy
    Verbose=1, ... % made true for local
    ValidationPatience=15,... %if validation loss goes without improving for 5 checks -> early stopping
    OutputNetwork="best-validation");
% removed %,...'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3)


disp('BiLSTM training started')
%Train lstm
[lstm, info_BiLSTM] = trainnet(trainingFeatures_RICA_Cell,trainLabels_Cell,netLSTM,"crossentropy",options);
% 
% allTrainLabels = [];
% for c = 1:length(trainLabels_Cell)
% 
%     allTrainLabels = [allTrainLabels; trainLabels_Cell{c}(:)]; 
% end
% labelCounts = countcats(allTrainLabels);
% totalWindows = sum(labelCounts);
% 
% 
% classWeights = totalWindows ./ (numClasses * labelCounts);
% 
% classWeights = single(classWeights'); 
% 
% disp('Class Weights (Alphabetical Order):');
% disp(classWeights);
% 
% weightedLoss = @(Y,T) crossentropy(Y, T, 'Weights', classWeights, 'WeightsFormat', 'C');
% 
% [lstm, info_BiLSTM] = trainnet(trainingFeatures_RICA_Cell, trainLabels_Cell, netLSTM, weightedLoss, options);
classNames = categories(trainLabels);
% 
% disp('Saving trained BiLSTM to checkpoint...');
% save('BiLSTM_Trained_Model.mat', 'lstm', 'classNames');
% disp('Checkpoint saved securely!');
% savig traininglogs
save('BiLSTM_training_log_test3.mat', 'info_BiLSTM');


% Predict validation set
disp('Predicting validation set...');
scores_validation = minibatchpredict(lstm, validationFeatures_RICA_Cell, InputDataFormats="CTB", UniformOutput=false);

predictedLabels_validation = [];
validationLabels_flat = [];

for i = 1:length(scores_validation)
    true_seq = validationLabels_Cell{i};
    true_seq = true_seq(:); 
    
    pred_scores = scores_validation{i};
    pred_seq = scores2label(pred_scores, classNames);
    pred_seq = pred_seq(:); 
    
    % strip any padding
    if length(pred_seq) > length(true_seq)
        pred_seq = pred_seq(1:length(true_seq));
    elseif length(pred_seq) < length(true_seq)
        true_seq = true_seq(1:length(pred_seq));
    end
    predictedLabels_validation = [predictedLabels_validation; pred_seq];
    validationLabels_flat = [validationLabels_flat; true_seq];
end

% Plot confusion matrix
% fig_val = figure('Visible','off'); 
% confusionchart(validationLabels_flat, predictedLabels_validation, Title="Validation Set");
% exportgraphics(fig_val, 'Val_BiLSTM_confusion.png','Resolution',300);


disp('Predicting test set...');
scores_test = minibatchpredict(lstm, testFeatures_RICA_Cell, InputDataFormats="CTB", UniformOutput=false);

predictedLabels_test = [];
testLabels_flat = [];

for i = 1:length(scores_test)
    true_seq = testLabels_Cell{i};
    true_seq = true_seq(:); 
    
    pred_scores = scores_test{i};
    pred_seq = scores2label(pred_scores, classNames);
    pred_seq = pred_seq(:); 
    
    if length(pred_seq) > length(true_seq)
        pred_seq = pred_seq(1:length(true_seq));
    elseif length(pred_seq) < length(true_seq)
        true_seq = true_seq(1:length(pred_seq));
    end
    
    predictedLabels_test = [predictedLabels_test; pred_seq];
    testLabels_flat = [testLabels_flat; true_seq];
end

% Plot confusion matrix
% fig_test = figure('Visible','off');
% confusionchart(testLabels_flat, predictedLabels_test, Title="Test Set");
% exportgraphics(fig_test, 'test_BiLSTM_confusion.png', 'Resolution',300);

%% Voting

predictedLabels_validationset_1PerCateg=[];
for i = 1:3:length(predictedLabels_validation)
    predictedLabels_validationset_1PerCateg = [predictedLabels_validationset_1PerCateg, mode(predictedLabels_validation (i:i+2))];  
end

validationlabel_1PerCateg=[];
for i = 1:3:length(validationLabels_flat)
    validationlabel_1PerCateg = [validationlabel_1PerCateg, validationLabels_flat(i)];  
end

% Plot confusion matrix 
% fig_vote_val = figure('Visible','off');
% confusionchart(validationlabel_1PerCateg(:),predictedLabels_validationset_1PerCateg(:), Title ="Final validation Confusion Matrix");
% exportgraphics(fig_vote_val, 'Voting_val_confusion.png', "Resolution",300);

true_labels_val = string(validationlabel_1PerCateg(:));
model_preds_val = string(predictedLabels_validationset_1PerCateg(:));
results_table = table(true_labels_val, model_preds_val, 'VariableNames', {'true_labels', 'model_preds'});
writetable(results_table, 'final_pred_val_test3.csv');

% Performance evaluation on the testset (voting per 4 image)

% There are three images for each window, since there were 3 axes (i.e., x, y, z) and magnitude.
% We do majority voting for every 3 images that belong to the same window


predictedLabels_testset_1PerCateg=[];
for i = 1:3:length(predictedLabels_test)
    predictedLabels_testset_1PerCateg = [predictedLabels_testset_1PerCateg, mode(predictedLabels_test (i:i+2))];  
end
% We also do take one of the true labeles to create the vectors, which
% shows th correct lables for each window.

testlabel_1PerCateg=[];
for i = 1:3:length(testLabels_flat)
    testlabel_1PerCateg = [testlabel_1PerCateg, testLabels_flat(i)];  
end

% Plot confusion matrix
% fig_vote_test = figure('Visible','off');
% confusionchart(testlabel_1PerCateg(:),predictedLabels_testset_1PerCateg(:),Title = "Test Confusion matrix");
% exportgraphics(fig_vote_test, 'voting_test_confusion.png','Resolution',300);

% exporting to Csv for analysis in python
true_labels = string(testlabel_1PerCateg(:));
model_preds = string(predictedLabels_testset_1PerCateg(:));
results_table = table(true_labels, model_preds, 'VariableNames', {'true_labels', 'model_preds'});
writetable(results_table, 'final_pred_test_test3.csv');