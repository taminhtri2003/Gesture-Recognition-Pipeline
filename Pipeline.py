
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import os
from sklearn.decomposition import PCA
from Pipeline_function import load_ninapro_db5_file, plot_emg_details, apply_filter, segment_and_extract_features, train_and_evaluate_model
# Constants
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Configuration Parameters ---
MAT_FILE_PATH = "S5_E2_A1.mat"  # !!! REPLACE WITH YOUR .MAT FILE PATH !!!

# EMG Processing Parameters
FS = None 
FILTER_TYPE = 'bandpass' 
LOWCUT = 20.0  
HIGHCUT = 95.0 
FILTER_ORDER = 4

# Windowing Parameters for Feature Extraction
WINDOW_DURATION_MS = 256 
WINDOW_OVERLAP_PERCENT = 50 

# Visualization Parameters
CHANNELS_FOR_VISUALIZATION = [0, 1, 7, 15] 
RMS_WINDOW_FOR_PLOT_MS = 50 

# Deep Learning Model Parameters
DL_EPOCHS = 50
DL_BATCH_SIZE = 32

if __name__ == '__main__':
    emg_raw, stimulus_raw, fs_loaded = load_ninapro_db5_file(MAT_FILE_PATH)
    if emg_raw is None: print("Exiting: data loading failure."); exit()
    FS = fs_loaded 

    print("\nPlotting Raw EMG (Before Filtering)...")
    plot_emg_details(emg_raw, stimulus_raw, FS, channels_to_plot=CHANNELS_FOR_VISUALIZATION,
                     title_suffix=" - Raw Signal (Before Filtering)", main_signal_label="Raw EMG")

    print(f"\nApplying {FILTER_TYPE} filter (Low:{LOWCUT}Hz, High:{HIGHCUT}Hz, Order:{FILTER_ORDER})...")
    emg_filtered = apply_filter(emg_raw, LOWCUT, HIGHCUT, FS, order=FILTER_ORDER, filter_type=FILTER_TYPE)
    
    emg_for_features = emg_raw 
    if emg_filtered is not None and emg_filtered.shape == emg_raw.shape:
        print("\nPlotting Filtered EMG (After Filtering)...")
        plot_emg_details(emg_filtered, stimulus_raw, FS, show_rms=True, rms_window_ms=RMS_WINDOW_FOR_PLOT_MS, 
                         channels_to_plot=CHANNELS_FOR_VISUALIZATION, title_suffix=" - Filtered Signal (After Filtering)",
                         main_signal_label="Filtered EMG")
        emg_for_features = emg_filtered
    else:
        print("Filter fail/mismatch. Using raw EMG for features. Skipping 'After Filtering' plot.")

    print(f"\nSegmenting & extracting features (Win:{WINDOW_DURATION_MS}ms, Overlap:{WINDOW_OVERLAP_PERCENT}%)...")
    # `extracted_labels` here are the original stimulus numbers (e.g., 1, 2, ...) for non-rest states
    features, extracted_labels, feature_names_list = segment_and_extract_features(
        emg_for_features, stimulus_raw, FS, WINDOW_DURATION_MS, WINDOW_OVERLAP_PERCENT)
    if features.size==0: print("No features extracted. Exit."); exit()
    print(f"Extracted {features.shape[0]}x{features.shape[1]} feats, {extracted_labels.shape[0]} labels. Unique: {np.unique(extracted_labels)}")
    if len(np.unique(extracted_labels))<2: print("Error: <2 classes for classification. Exit."); exit()

    scaler=StandardScaler(); features_scaled=scaler.fit_transform(features); print("Features scaled.")
    
    processed_model_features = features_scaled # No PCA
    
    if processed_model_features.shape[0]==0: print("Error: No processed features. Exit."); exit()

    X_tr,X_te,y_tr,y_te=train_test_split(processed_model_features,extracted_labels,test_size=0.3,random_state=42,stratify=extracted_labels)
    print(f"\nData split: Train {X_tr.shape[0]}, Test {X_te.shape[0]}.")
    
    # Create target_names_map based on all unique labels found in the extracted_labels
    # This map is used for display purposes in classification reports
    all_unique_gestures_in_data = sorted(np.unique(extracted_labels))
    target_names_map={l:f"Gesture {l}" for l in all_unique_gestures_in_data}

    results=[]
    # For DL model, pass all unique labels from the dataset to help define num_classes consistently
    all_labels_for_dl_model = extracted_labels 
    
    models_to_run = ['SVM','RandomForest','KNN', 'SimpleDL']
    for model_name_iter in models_to_run:
        model_run_results=train_and_evaluate_model(X_tr,y_tr,X_te,y_te,model_name_iter,
                                        fn=feature_names_list, # `fn` is for feature names
                                        tn=target_names_map,  # `tn` is for target names map
                                        all_labels_for_dl=all_labels_for_dl_model if model_name_iter == 'SimpleDL' else None)
        results.append(model_run_results)

    print("\n--- Model Comparison ---")
    results_df=pd.DataFrame(results)
    # Ensure columns exist before trying to select them, especially 'importances' which is DL-specific as None
    cols_to_show = ['model', 'accuracy', 'f1_macro']
    # if 'importances' in results_df.columns: # Not useful to print importances here
    #     cols_to_show.append('importances')
    print(results_df[cols_to_show].sort_values(by='f1_macro',ascending=False))
    
    results_df.set_index('model')[['accuracy', 'f1_macro']].plot(kind='bar',figsize=(12,7)) # Wider plot
    plt.title('Model Performance Comparison');plt.ylabel('Score');plt.xticks(rotation=45, ha='right');plt.tight_layout();plt.show()
    print("\nPipeline finished.")
