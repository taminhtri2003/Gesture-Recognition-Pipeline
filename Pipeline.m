% MATLAB Script for EMG Signal Processing and Classification (Ninapro DB5)
% Enhanced with more features, feature selection, and feature-output correlation viz

% --- 0. Clear Workspace and Command Window ---
clear; clc; close all;

disp('Starting EMG Signal Processing Pipeline (Enhanced with Correlation Viz)...');

% --- Parameters ---
% Feature Extraction
wamp_threshold = 0.05; % Threshold for Willison Amplitude (adjust based on signal magnitude)
myop_threshold = 0.05; % Threshold for Myopulse Percentage Rate (adjust based on signal magnitude)

% Feature Selection
num_top_features_to_select = 30; % Number of top features to select after ranking
num_features_for_boxplot_viz = 6; % Number of top selected features to visualize with boxplots

% --- 1. Load Data ---
disp('--- Step 1: Loading Data ---');
try
    [fileName, pathName] = uigetfile('*.mat', 'Select the Ninapro DB5 .mat file');
    if isequal(fileName,0), disp('User selected Cancel'); return; end
    fullPath = fullfile(pathName, fileName);
    disp(['User selected ', fullPath]);
    data = load(fullPath);
    disp('Data loaded successfully.');
catch ME
    disp('Error loading .mat file:'); disp(ME.message); return;
end

emg_signals = double(data.emg);
labels = double(data.stimulus);
if isfield(data, 'restimulus'), labels = double(data.restimulus); disp('Using ''restimulus''.'); end
repetitions = double(data.repetition);
if isfield(data, 'rerepetition'), repetitions = double(data.rerepetition); disp('Using ''rerepetition''.'); end
sampling_frequency = data.frequency;
num_channels = size(emg_signals, 2);
num_samples = size(emg_signals, 1);

disp(['Fs: ', num2str(sampling_frequency), ' Hz, Channels: ', num2str(num_channels), ', Samples: ', num2str(num_samples)]);
disp(['Unique Labels: ', mat2str(unique(labels))]);

% --- 1.1. Visualize Raw EMG Signal (Selected Channels) ---
disp('--- Step 1.1: Visualizing Raw EMG Data ---');
if num_channels >= 4
    channels_to_plot = [1, 2, 3, 4];
    channels_to_plot = channels_to_plot(channels_to_plot <= num_channels);
    if length(channels_to_plot) > 4, channels_to_plot = channels_to_plot(1:4); end
    if isempty(channels_to_plot) && num_channels > 0, channels_to_plot = 1;
    elseif isempty(channels_to_plot) && num_channels == 0, disp('No EMG channels.'); return; end
else
    channels_to_plot = 1:num_channels;
end
disp(['Plotting raw for channels: ', mat2str(channels_to_plot)]);
time_vector = (0:num_samples-1) / sampling_frequency;
figure('Name', 'Raw EMG - Selected Channels', 'NumberTitle', 'off', 'WindowState', 'maximized');
num_plot_channels = length(channels_to_plot);
unique_stimuli = unique(labels(labels~=0));
plot_colors = lines(max(1,length(unique_stimuli))); % Renamed 'colors' to 'plot_colors'

for plot_idx = 1:num_plot_channels
    current_channel = channels_to_plot(plot_idx);
    subplot(num_plot_channels, 1, plot_idx);
    plot(time_vector, emg_signals(:, current_channel));
    hold on;
    for i = 1:length(unique_stimuli)
        stimulus_val = unique_stimuli(i);
        diff_stimulus = diff([0; labels == stimulus_val; 0]);
        start_indices = find(diff_stimulus == 1);
        end_indices = find(diff_stimulus == -1) -1;
        for k = 1:length(start_indices)
            if start_indices(k) <= num_samples && end_indices(k) <= num_samples && start_indices(k) <= end_indices(k)
                x_coords = time_vector([start_indices(k), end_indices(k), end_indices(k), start_indices(k)]);
                y_coords = [min(emg_signals(:, current_channel)), min(emg_signals(:, current_channel)), max(emg_signals(:, current_channel)), max(emg_signals(:, current_channel))];
                patch(x_coords, y_coords, plot_colors(i,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', ['Stim ' num2str(stimulus_val)]);
            end
        end
    end
    hold off; ylabel('Amplitude'); title(['Raw EMG Ch ', num2str(current_channel)]); grid on;
    if plot_idx == 1, legend('show', 'Location', 'bestoutside'); end
    if plot_idx < num_plot_channels, set(gca,'xticklabel',[]); else, xlabel('Time (s)'); end
end
sgtitle('Raw EMG Signals with Stimulus Labels');
disp('Raw EMG viz complete.');

% --- 2. Filter Design and Application ---
disp('--- Step 2: Filtering EMG Data ---');
bp_low_cutoff = 20; bp_high_cutoff = min(450, sampling_frequency/2 - 1); bp_filter_order = 4;
[b_bp, a_bp] = butter(bp_filter_order, [bp_low_cutoff bp_high_cutoff] / (sampling_frequency/2), 'bandpass');
emg_filtered_bp = zeros(size(emg_signals));
for i = 1:num_channels, emg_filtered_bp(:, i) = filtfilt(b_bp, a_bp, emg_signals(:, i)); end
disp('Band-pass filtering complete.');

notch_freq = 50; bandwidth = 2;
wo_notch = notch_freq / (sampling_frequency/2); bw_notch_param = wo_notch / (notch_freq / bandwidth);
if wo_notch > 0 && wo_notch < 1
    [b_notch, a_notch] = iirnotch(wo_notch, bw_notch_param);
    emg_filtered_final = zeros(size(emg_filtered_bp));
    for i = 1:num_channels, emg_filtered_final(:, i) = filtfilt(b_notch, a_notch, emg_filtered_bp(:, i)); end
    disp(['Notch filtering @ ', num2str(notch_freq), ' Hz complete.']);
else
    disp(['Notch freq ', num2str(notch_freq), ' Hz invalid. Skipping.']);
    emg_filtered_final = emg_filtered_bp;
end

% --- 2.3. Visualize Filtered EMG Signal ---
disp('--- Step 2.3: Visualizing Filtered EMG Data ---');
figure('Name', 'Filtered EMG - Selected Channels', 'NumberTitle', 'off', 'WindowState', 'maximized');
for plot_idx = 1:num_plot_channels
    current_channel = channels_to_plot(plot_idx);
    subplot(num_plot_channels, 1, plot_idx);
    plot(time_vector, emg_filtered_final(:, current_channel));
    hold on;
    for i = 1:length(unique_stimuli)
        stimulus_val = unique_stimuli(i);
        diff_stimulus = diff([0; labels == stimulus_val; 0]);
        start_indices = find(diff_stimulus == 1);
        end_indices = find(diff_stimulus == -1) -1;
        for k = 1:length(start_indices)
             if start_indices(k) <= num_samples && end_indices(k) <= num_samples && start_indices(k) <= end_indices(k)
                x_coords = time_vector([start_indices(k), end_indices(k), end_indices(k), start_indices(k)]);
                y_coords = [min(emg_filtered_final(:, current_channel)), min(emg_filtered_final(:, current_channel)), max(emg_filtered_final(:, current_channel)), max(emg_filtered_final(:, current_channel))];
                patch(x_coords, y_coords, plot_colors(i,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', ['Stim ' num2str(stimulus_val)]);
            end
        end
    end
    hold off; ylabel('Amplitude'); title(['Filtered EMG Ch ', num2str(current_channel)]); grid on;
    if plot_idx == 1, legend('show', 'Location', 'bestoutside'); end
    if plot_idx < num_plot_channels, set(gca,'xticklabel',[]); else, xlabel('Time (s)'); end
end
sgtitle('Filtered EMG Signals with Stimulus Labels');
disp('Filtered EMG viz complete.');

% --- 3. Feature Extraction (Expanded) ---
disp('--- Step 3: Feature Extraction (Expanded) ---');
window_size_sec = 0.2; window_overlap_perc = 0.5;
window_length = round(window_size_sec * sampling_frequency);
window_overlap = round(window_length * window_overlap_perc);
window_step = window_length - window_overlap;
disp(['WinLen: ',num2str(window_length),' samp, Overlap: ',num2str(window_overlap),' samp, Step: ',num2str(window_step),' samp']);

features_all = [];
labels_all_features = [];
repetition_all_features = [];
feature_names = {}; % Store names for plotting

% Generate feature names once
td_feature_short_names = {'MAV', 'RMS', 'ZC', 'SSC', 'WL', 'VAR', 'IEMG', 'SKEW', 'KURT', 'WAMP', 'MYOP'};
fd_feature_short_names = {'MNF', 'MDF', 'PKF', 'TTP'};
for ch_idx = 1:num_channels
    for f_idx = 1:length(td_feature_short_names)
        feature_names{end+1} = [td_feature_short_names{f_idx} '_Ch' num2str(ch_idx)];
    end
    for f_idx = 1:length(fd_feature_short_names)
        feature_names{end+1} = [fd_feature_short_names{f_idx} '_Ch' num2str(ch_idx)];
    end
end


active_indices = find(labels ~= 0);
if isempty(active_indices), disp('No active movement labels. Cannot extract features.'); return; end
diff_active = diff([0; (labels(active_indices) ~= 0); 0]);
start_active_blocks_relative = find(diff_active == 1);
end_active_blocks_relative = find(diff_active == -1) -1;

if isempty(start_active_blocks_relative) || isempty(end_active_blocks_relative)
     if ~isempty(active_indices) % Handle case where all active_indices form one block
        start_active_blocks = active_indices(1); 
        end_active_blocks = active_indices(end);
     else
        disp('No contiguous active blocks found.'); return;
     end
else
    start_active_blocks = active_indices(start_active_blocks_relative);
    end_active_blocks = active_indices(end_active_blocks_relative);
end

disp(['Found ', num2str(length(start_active_blocks)), ' active movement blocks.']);

for block_idx = 1:length(start_active_blocks)
    block_start = start_active_blocks(block_idx); block_end = end_active_blocks(block_idx);
    if block_start > length(labels) || block_end > length(labels) || block_start > block_end || block_start < 1 || block_end < 1
        disp(['Skipping invalid block: ', num2str(block_start), '-', num2str(block_end)]); continue;
    end
    current_block_emg = emg_filtered_final(block_start:block_end, :);
    current_block_labels = labels(block_start:block_end);
    current_block_repetitions = repetitions(block_start:block_end);

    for i = 1:window_step:(size(current_block_emg, 1) - window_length + 1)
        window_emg = current_block_emg(i : i+window_length-1, :);
        idx_end_window_labels = min(i+window_length-1, length(current_block_labels));
        window_label_segment = current_block_labels(i : idx_end_window_labels);
        window_repetition_segment = current_block_repetitions(i : idx_end_window_labels);
        if isempty(window_label_segment), continue; end
        window_label = mode(window_label_segment);
        window_repetition = mode(window_repetition_segment);

        if length(unique(window_label_segment)) == 1 && window_label ~= 0
            current_features_vector = [];
            for ch = 1:num_channels
                segment = window_emg(:, ch);
                % Time-Domain Features
                mav = mean(abs(segment));
                rms = sqrt(mean(segment.^2));
                zc = sum(diff(sign(segment)) ~= 0);
                ssc = sum(diff(sign(diff(segment))) ~= 0);
                wl = sum(abs(diff(segment)));
                var_feat = var(segment);
                iemg = sum(abs(segment));
                skew_feat = skewness(segment);
                kurt_feat = kurtosis(segment);
                wamp = sum(abs(diff(segment)) > wamp_threshold);
                myop = sum(abs(segment) > myop_threshold) / window_length;

                td_feats = [mav, rms, zc, ssc, wl, var_feat, iemg, skew_feat, kurt_feat, wamp, myop];
                
                % Frequency-Domain Features
                NFFT = 2^nextpow2(window_length); 
                Y = fft(segment, NFFT)/window_length;
                f_axis = sampling_frequency/2 * linspace(0,1,NFFT/2+1); % Renamed 'f' to 'f_axis'
                psd = abs(Y(1:NFFT/2+1)).^2; 
                
                current_mnf = sum(f_axis .* psd') / sum(psd); 
                cs_psd = cumsum(psd);
                mdf_idx = find(cs_psd >= cs_psd(end)/2, 1, 'first'); 
                current_mdf = f_axis(mdf_idx); 
                [~, pkf_idx] = max(psd); 
                current_pkf = f_axis(pkf_idx);
                current_ttp = sum(psd); 

                if sum(psd) < eps % Check for near-zero power
                    current_mnf = 0; current_mdf = 0; current_pkf = 0; % ttp is already ~0
                end
                
                fd_feats = [current_mnf, current_mdf, current_pkf, current_ttp];
                current_features_vector = [current_features_vector, td_feats, fd_feats];
            end
            features_all = [features_all; current_features_vector];
            labels_all_features = [labels_all_features; window_label];
            repetition_all_features = [repetition_all_features; window_repetition];
        end
    end
end

if isempty(features_all), disp('No features extracted.'); return; end
disp(['Feature extraction complete. Total vectors: ', num2str(size(features_all, 1)), ...
      ', Features/vector: ', num2str(size(features_all, 2))]);

% --- 3.1. Feature Selection ---
disp('--- Step 3.1: Feature Selection ---');
selected_feature_names = {}; % Initialize to prevent errors if selection is skipped

if size(features_all, 1) <= size(features_all, 2) || size(features_all,1) < 10 
    disp('Warning: Not enough samples or too many features for reliable feature ranking. Using all features.');
    selected_feature_indices = 1:size(features_all,2);
    features_selected = features_all;
    if ~isempty(feature_names)
        selected_feature_names = feature_names;
    else
        selected_feature_names = arrayfun(@(x) ['Feat' num2str(x)], 1:size(features_all,2), 'UniformOutput', false);
    end
else
    try
        % fsrftest requires Statistics and Machine Learning Toolbox
        if license('test', 'Statistics_Toolbox')
            [~, scores] = fsrftest(features_all, labels_all_features); 
            [sorted_scores, sorted_idx_fs] = sort(scores, 'descend');
            
            actual_num_to_select = min(num_top_features_to_select, length(sorted_idx_fs));
            selected_feature_indices = sorted_idx_fs(1:actual_num_to_select);
            features_selected = features_all(:, selected_feature_indices);
            
            if ~isempty(feature_names)
                selected_feature_names = feature_names(selected_feature_indices);
            else 
                 selected_feature_names = arrayfun(@(x) ['Feat' num2str(x)], selected_feature_indices, 'UniformOutput', false);
            end

            disp(['Selected top ', num2str(actual_num_to_select), ' features using fsrftest.']);

            figure('Name', 'Feature Importance Scores (F-test)', 'NumberTitle', 'off', 'WindowState', 'maximized');
            num_scores_to_plot = min(50, length(sorted_scores));
            bar(sorted_scores(1:num_scores_to_plot)); 
            xlabel('Feature Rank');
            ylabel('F-Score');
            title(['Top ',num2str(num_scores_to_plot),' Feature Importance Scores (F-test)']);
            grid on;
            if ~isempty(feature_names) && actual_num_to_select > 0 && num_scores_to_plot > 0
                 xticklabels(feature_names(sorted_idx_fs(1:num_scores_to_plot)));
                 xtickangle(45);
                 set(gca, 'TickLabelInterpreter', 'none'); % To display underscores correctly
            end
            disp('Feature importance plot generated.');
        else
            disp('Statistics and Machine Learning Toolbox not found. Skipping fsrftest. Using all features.');
            selected_feature_indices = 1:size(features_all,2);
            features_selected = features_all;
            if ~isempty(feature_names), selected_feature_names = feature_names;
            else, selected_feature_names = arrayfun(@(x) ['Feat' num2str(x)],1:size(features_all,2),'UniformOutput',false); end
        end
    catch ME_fs
        disp('Error during feature selection:'); disp(ME_fs.message);
        disp('Using all features due to error.');
        selected_feature_indices = 1:size(features_all,2);
        features_selected = features_all;
        if ~isempty(feature_names), selected_feature_names = feature_names;
        else, selected_feature_names = arrayfun(@(x) ['Feat' num2str(x)],1:size(features_all,2),'UniformOutput',false); end
    end
end
disp(['Using ', num2str(size(features_selected,2)), ' features for classification.']);


% --- 3.2. Visualize Selected Feature Distributions by Class (NEW SECTION) ---
disp('--- Step 3.2: Visualizing Selected Feature Distributions by Class ---');
if ~isempty(features_selected) && ~isempty(labels_all_features) && ~isempty(selected_feature_names) && num_features_for_boxplot_viz > 0
    num_to_plot_boxplot = min(num_features_for_boxplot_viz, size(features_selected, 2));
    if num_to_plot_boxplot > 0
        figure('Name', 'Selected Feature Distributions by Class', 'NumberTitle', 'off', 'WindowState', 'maximized');
        % Determine subplot layout (e.g., 2x3 or 3x2 for 6 features)
        nrows_subplot = ceil(sqrt(num_to_plot_boxplot));
        ncols_subplot = ceil(num_to_plot_boxplot / nrows_subplot);

        for feat_idx = 1:num_to_plot_boxplot
            subplot(nrows_subplot, ncols_subplot, feat_idx);
            boxplot(features_selected(:, feat_idx), labels_all_features(train_idx), 'Labels', unique(labels_all_features(train_idx))); % Use train_idx to match training data context if desired
            % Or use all labels: boxplot(features_selected(:, feat_idx), labels_all_features, 'Labels', unique(labels_all_features));
            title(selected_feature_names{feat_idx}, 'Interpreter', 'none'); % 'none' interpreter for underscores
            ylabel('Feature Value');
            xlabel('Class Label');
            grid on;
        end
        sgtitle(['Distributions of Top ', num2str(num_to_plot_boxplot), ' Selected Features by Class']);
        disp(['Boxplot visualization for top ', num2str(num_to_plot_boxplot), ' selected features generated.']);
    else
        disp('Not enough selected features to generate boxplots or num_features_for_boxplot_viz is 0.');
    end
else
    disp('Skipping feature distribution boxplots (no selected features, labels, or names available, or num_features_for_boxplot_viz is 0).');
end


% --- 4. Machine Learning Models ---
disp('--- Step 4: Machine Learning Models ---');
% --- 4.1. Data Splitting (Train/Test) ---
unique_reps = unique(repetition_all_features);
if length(unique_reps) < 2
    disp('Not enough unique reps. Using CV or all data for train/test.');
    if length(unique_reps) == 1
        disp('Only 1 rep. Using all data for train/test (not ideal).');
        train_idx = true(size(labels_all_features)); test_idx = true(size(labels_all_features));
    else % No repetition data or other issue
        if isempty(labels_all_features)
            disp('labels_all_features is empty. Cannot proceed with CV split.'); return;
        end
        cv_partition = cvpartition(labels_all_features, 'KFold', 5);
        train_idx = training(cv_partition, 1); test_idx = test(cv_partition, 1);
        disp('Using 5-fold CV split (1st fold for train/test).');
    end
    train_reps_disp = 'N/A (CV or single rep)'; test_reps_disp = 'N/A (CV or single rep)';
else
    num_train_reps = floor(0.7 * length(unique_reps));
    if num_train_reps == 0 && length(unique_reps) >= 1, num_train_reps = 1; end % Ensure at least one training rep if reps exist
    
    if num_train_reps == 0 
         disp('Fallback: Using all data for train/test due to rep split issue.');
         train_idx = true(size(labels_all_features)); test_idx = true(size(labels_all_features));
         train_reps = unique_reps; test_reps = unique_reps;
    else
        train_reps = unique_reps(1:num_train_reps);
        test_reps_candidate = unique_reps(num_train_reps+1:end);
        if isempty(test_reps_candidate) && length(unique_reps) > num_train_reps % If 70% rule left no test reps but more reps exist
             test_reps = unique_reps(end); % assign last rep to test
        elseif isempty(test_reps_candidate) % All reps already in train_reps
            test_reps = train_reps; % Test on train if only one group of reps
            disp('Warning: All available repetitions assigned to training. Testing on training data for repetition split.');
        else
            test_reps = test_reps_candidate;
        end
        % Ensure train_reps and test_reps are not identical if possible
        if isequal(sort(train_reps), sort(test_reps)) && length(unique_reps) > 1
            if length(train_reps) > 1
                test_reps = train_reps(end); % take one from train for test
                train_reps = train_reps(1:end-1);
            elseif length(test_reps) > 1 % Should not happen if train_reps had only one
                train_reps = test_reps(end);
                test_reps = test_reps(1:end-1);
            else
                 disp('Warning: Cannot make train and test repetitions distinct with current logic.');
            end
        end
    end
    train_idx = ismember(repetition_all_features, train_reps);
    test_idx = ismember(repetition_all_features, test_reps);
    
    if ~any(test_idx) && any(train_idx) % Ensure test set is not empty if train set exists
        disp('Warning: Test set empty after rep split. Assigning last available rep to test.');
        available_test_reps = setdiff(unique_reps, train_reps);
        if ~isempty(available_test_reps)
            test_reps = available_test_reps(1); % Take the first available unassigned rep
            test_idx = ismember(repetition_all_features, test_reps);
        else % All reps used for training, test on training data
            test_idx = train_idx; test_reps = train_reps; % Test on training data
            disp('Critical Warning: All reps in train. Testing on training data.');
        end
    elseif ~any(train_idx) && ~isempty(labels_all_features)
        disp('Warning: Training set empty after rep split. Using all data for training.');
        train_idx = true(size(labels_all_features));
        if ~any(test_idx) % If test is also empty
            test_idx = train_idx;
            disp('Also using all data for testing.');
        end
    end
    train_reps_disp = mat2str(train_reps); test_reps_disp = mat2str(test_reps);
end
disp(['Train Reps: ', train_reps_disp, ', Test Reps: ', test_reps_disp]);

X_train = features_selected(train_idx, :);
y_train = labels_all_features(train_idx);
X_test = features_selected(test_idx, :);
y_test = labels_all_features(test_idx);

if isempty(X_train) || isempty(y_train), disp('Training set empty. Cannot proceed.'); return; end
if isempty(X_test) || isempty(y_test)
    disp('Test set empty. Forcing test on train data for script continuation.');
    X_test = X_train; y_test = y_train;
    disp('WARNING: Testing on training data. Results will not reflect generalization performance.');
end
disp(['Train Samp: ', num2str(size(X_train,1)), ', Test Samp: ', num2str(size(X_test,1))]);

% --- 4.2. Model Training and Prediction ---
model_names_list = {}; accuracies_list = []; predictions_all_models_list = {};

% SVM
disp('Training SVM...'); try
    if license('test', 'Statistics_Toolbox')
        svm_model = fitcecoc(X_train, y_train, 'Learners', 'svm', 'Coding', 'onevsone');
        y_pred_svm = predict(svm_model, X_test);
        acc_svm = sum(y_pred_svm == y_test) / length(y_test) * 100;
        disp(['SVM Acc: ', num2str(acc_svm), '%']);
        model_names_list{end+1} = 'SVM'; accuracies_list(end+1) = acc_svm; predictions_all_models_list{end+1} = y_pred_svm;
    else disp('SVM training skipped: Statistics Toolbox license not found.'); end
catch ME_svm, disp('Error SVM:'); disp(ME_svm.message); end

% k-NN
disp('Training k-NN...'); try
    if license('test', 'Statistics_Toolbox')
        knn_model = fitcknn(X_train, y_train, 'NumNeighbors', 5, 'Distance', 'euclidean');
        y_pred_knn = predict(knn_model, X_test);
        acc_knn = sum(y_pred_knn == y_test) / length(y_test) * 100;
        disp(['k-NN Acc: ', num2str(acc_knn), '%']);
        model_names_list{end+1} = 'k-NN'; accuracies_list(end+1) = acc_knn; predictions_all_models_list{end+1} = y_pred_knn;
    else disp('k-NN training skipped: Statistics Toolbox license not found.'); end
catch ME_knn, disp('Error k-NN:'); disp(ME_knn.message); end

% LDA
disp('Training LDA...'); try
    if license('test', 'Statistics_Toolbox')
        if rank(X_train) < min(size(X_train)) && rank(X_train) < length(unique(y_train)) -1
            disp('Warning: LDA data rank deficient. Using pseudoLinear.');
            lda_model = fitcdiscr(X_train, y_train, 'DiscrimType', 'pseudoLinear');
        else
            lda_model = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
        end
        y_pred_lda = predict(lda_model, X_test);
        acc_lda = sum(y_pred_lda == y_test) / length(y_test) * 100;
        disp(['LDA Acc: ', num2str(acc_lda), '%']);
        model_names_list{end+1} = 'LDA'; accuracies_list(end+1) = acc_lda; predictions_all_models_list{end+1} = y_pred_lda;
    else disp('LDA training skipped: Statistics Toolbox license not found.'); end
catch ME_lda, disp('Error LDA:'); disp(ME_lda.message); end

% Random Forest
disp('Training Random Forest...'); try
    if license('test', 'Statistics_Toolbox')
        rf_model = fitcensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 50, 'Learners', 'Tree');
        y_pred_rf = predict(rf_model, X_test);
        acc_rf = sum(y_pred_rf == y_test) / length(y_test) * 100;
        disp(['RF Acc: ', num2str(acc_rf), '%']);
        model_names_list{end+1} = 'Random Forest'; accuracies_list(end+1) = acc_rf; predictions_all_models_list{end+1} = y_pred_rf;
    else disp('RF training skipped: Statistics Toolbox license not found.'); end
catch ME_rf, disp('Error RF:'); disp(ME_rf.message); end

% --- 5. Compare Results Between Models ---
disp('--- Step 5: Comparing Model Results ---');
if ~isempty(accuracies_list)
    figure('Name', 'Model Comparison - Accuracy', 'NumberTitle', 'off', 'WindowState', 'maximized');
    try
        bar_handle = bar(accuracies_list);
        if ~isempty(bar_handle) && isprop(bar_handle, 'XEndPoints') % Check for property
            xtips = bar_handle.XEndPoints; ytips = bar_handle.YEndPoints;
            labels_bar_text = string(round(accuracies_list,2)); % Renamed 'labels_bar'
            text(xtips,ytips,labels_bar_text,'HorizontalAlignment','center','VerticalAlignment','bottom');
        end
    catch, bar(accuracies_list); end 
    set(gca, 'XTickLabel', model_names_list, 'XTick', 1:length(model_names_list));
    ylabel('Accuracy (%)'); title('Comparison of ML Model Accuracies'); grid on;
    if ~isempty(accuracies_list), ylim([0 max([105, max(accuracies_list)+10])]); else, ylim([0 105]); end
    disp('Model Accuracies:');
    for i=1:length(model_names_list), disp([model_names_list{i}, ': ', num2str(accuracies_list(i)), '%']); end
else
    disp('No models successfully trained/tested to compare accuracies.');
end

% --- 6. Evaluate Performance ---
disp('--- Step 6: Performance Evaluation ---');
if ~isempty(predictions_all_models_list) && ~isempty(y_test)
    eval_model_idx = find(strcmp(model_names_list, 'SVM')); % Default to SVM
    if isempty(eval_model_idx) && ~isempty(model_names_list), eval_model_idx = 1; end % Fallback to first model

    if ~isempty(eval_model_idx) && eval_model_idx <= length(predictions_all_models_list)
        chosen_model_name_eval = model_names_list{eval_model_idx};
        y_pred_chosen_eval = predictions_all_models_list{eval_model_idx};
        disp(['Evaluating: ', chosen_model_name_eval]);
        figure('Name', ['CM - ', chosen_model_name_eval], 'NumberTitle', 'off', 'WindowState', 'maximized');
        try
            if license('test', 'Statistics_Toolbox')
                cm_obj = confusionchart(y_test, y_pred_chosen_eval);
                cm_obj.Title = ['Confusion Matrix for ', chosen_model_name_eval];
                cm_obj.RowSummary = 'row-normalized'; cm_obj.ColumnSummary = 'column-normalized';
            else
                disp('Confusion chart skipped: Statistics Toolbox license not found. Displaying numeric matrix.');
                confmat_fallback = confusionmat(y_test, y_pred_chosen_eval); disp(confmat_fallback);
            end
        catch ME_cm, disp('CM chart error:'); disp(ME_cm.message); confmat_fallback = confusionmat(y_test, y_pred_chosen_eval); disp(confmat_fallback); end
        
        conf_matrix_num = confusionmat(y_test, y_pred_chosen_eval);
        disp(['ConfMat (numeric) for ', chosen_model_name_eval, ':']); disp(conf_matrix_num);

        eval_class_labels_unique = union(unique(y_test), unique(y_pred_chosen_eval));
        num_classes_eval_unique = length(eval_class_labels_unique);
        precision_vals = zeros(1, num_classes_eval_unique); recall_vals = zeros(1, num_classes_eval_unique); f1_scores_vals = zeros(1, num_classes_eval_unique);

        for k_idx = 1:num_classes_eval_unique
            cls_lbl = eval_class_labels_unique(k_idx);
            tp = sum(y_pred_chosen_eval == cls_lbl & y_test == cls_lbl);
            fp = sum(y_pred_chosen_eval == cls_lbl & y_test ~= cls_lbl);
            fn = sum(y_pred_chosen_eval ~= cls_lbl & y_test == cls_lbl);
            precision_vals(k_idx) = tp / max(1e-9,(tp + fp)); 
            recall_vals(k_idx) = tp / max(1e-9,(tp + fn));    
            f1_scores_vals(k_idx) = 2 * (precision_vals(k_idx) * recall_vals(k_idx)) / max(1e-9,(precision_vals(k_idx) + recall_vals(k_idx)));
            disp(['Class ',num2str(cls_lbl),': P=',num2str(precision_vals(k_idx)*100,'%.2f'),'%, R=',num2str(recall_vals(k_idx)*100,'%.2f'),'%, F1=',num2str(f1_scores_vals(k_idx)*100,'%.2f'),'%']);
        end
        valid_f1_scores = f1_scores_vals(isfinite(f1_scores_vals) & f1_scores_vals > 0);
        if isempty(valid_f1_scores), mean_f1 = 0; else, mean_f1 = mean(valid_f1_scores); end
        disp(['Macro-Avg F1 for ',chosen_model_name_eval,': ',num2str(mean_f1*100,'%.2f'),'%']);
    else
        disp('No specific model selected/found for detailed evaluation.');
    end
else
    disp('No model predictions available or y_test is empty for evaluation.');
end

disp('--- Pipeline Finished ---');
