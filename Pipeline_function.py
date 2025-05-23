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
import warnings

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=RuntimeWarning) # For potential NaNs in feature extraction if signal is flat
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow INFO messages



# --- 1. Load Data ---
def load_ninapro_db5_file(file_path):
    """Loads data from a NinaPro DB5 .mat file."""
    try:
        mat_data = scipy.io.loadmat(file_path)
        print("Successfully loaded MAT file.")
        print("Available keys in MAT file:", mat_data.keys())
        emg = mat_data.get('emg')
        stimulus = mat_data.get('stimulus')
        global FS
        FS = mat_data.get('frequency', [[200]])[0,0] 
        if emg is None or stimulus is None: raise ValueError("EMG/stimulus data not found.")
        if stimulus.ndim > 1: stimulus = stimulus.ravel()
        min_len = min(emg.shape[0], stimulus.shape[0])
        if emg.shape[0] != stimulus.shape[0]:
            print(f"Warning: EMG/stimulus length mismatch. Truncating to {min_len}.")
            emg, stimulus = emg[:min_len, :], stimulus[:min_len]
        print(f"EMG: {emg.shape}, Stimulus: {stimulus.shape}, FS: {FS} Hz")
        unique_gestures = np.unique(stimulus)
        active_gestures = unique_gestures[unique_gestures != 0]
        print(f"Unique stimuli: {unique_gestures}, Active gestures: {active_gestures}")
        if len(active_gestures) < 2: print("Warning: <2 active gestures.")
        return emg, stimulus, FS
    except FileNotFoundError: print(f"Error: File not found {file_path}"); return None,None,None
    except Exception as e: print(f"Error loading MAT: {e}"); return None,None,None

# --- Helper function for RMS calculation ---
def calculate_moving_rms(signal, fs, window_ms):
    if signal is None or len(signal) == 0: return np.array([])
    n_samples = int(window_ms / 1000 * fs) 
    if n_samples <= 0 or n_samples > len(signal): return np.zeros_like(signal)
    kernel = np.ones(n_samples)/n_samples
    mean_sq = np.convolve(signal**2, kernel, mode='full')
    start, end = (n_samples-1)//2, (n_samples-1)//2 + len(signal)
    rms_val = np.sqrt(mean_sq[start:end])
    if len(rms_val) < len(signal): rms_val = np.pad(rms_val, (0, len(signal)-len(rms_val)), 'edge')
    elif len(rms_val) > len(signal): rms_val = rms_val[:len(signal)]
    return rms_val

# --- 2. Visualize EMG Signal ---
def plot_emg_details(emg_signal_data, stimulus_labels, fs, show_rms=False, 
                     rms_window_ms=50, channels_to_plot=None, title_suffix="",
                     main_signal_label="EMG Signal"):
    if emg_signal_data is None or stimulus_labels is None: print("Missing data for plotting."); return
    num_samples, num_ch_avail = emg_signal_data.shape
    time_vec = np.arange(num_samples) / fs
    if channels_to_plot is None or not channels_to_plot: channels_to_plot = list(range(min(num_ch_avail, 2)))
    else: channels_to_plot = [ch for ch in channels_to_plot if ch < num_ch_avail]
    if not channels_to_plot: print("No valid channels to plot."); return

    plt.figure(figsize=(18, 3.5 * len(channels_to_plot)))
    for i, ch_idx in enumerate(channels_to_plot):
        ax1 = plt.subplot(len(channels_to_plot), 1, i + 1)
        ch_data = emg_signal_data[:, ch_idx]
        color = 'darkorange' if 'Filtered' in main_signal_label else 'blue'
        ax1.plot(time_vec, ch_data, label=main_signal_label, color=color, alpha=0.8, lw=1.5)
        if show_rms:
            rms_vals = calculate_moving_rms(ch_data, fs, rms_window_ms)
            if rms_vals.size > 0:
                rms_base_label = main_signal_label.split()[0] if main_signal_label else "Signal"
                ax1.plot(time_vec, rms_vals, label=f'RMS of {rms_base_label} ({rms_window_ms}ms)', color='red', ls='--', lw=1.5)
        ax1.set_title(f'EMG Channel {ch_idx + 1}'); ax1.set_ylabel('Amplitude')
        ax1.legend(loc='upper left', fontsize='small'); ax1.grid(True, ls=':', alpha=0.6)
        ax2 = ax1.twinx()
        ax2.plot(time_vec, stimulus_labels[:num_samples], drawstyle='steps-post', label='Stimulus', color='green', alpha=0.7)
        ax2.set_ylabel('Stimulus ID', color='green'); ax2.tick_params(axis='y', labelcolor='green')
        unique_stim = np.unique(stimulus_labels)
        if len(unique_stim)>0: 
            ax2.set_yticks(unique_stim)
            min_s, max_s = min(unique_stim), max(unique_stim)
            ax2.set_ylim(min_s - (0.5 if min_s==max_s else (max_s-min_s)*0.1+0.1), 
                         max_s + (0.5 if min_s==max_s else (max_s-min_s)*0.1+0.1) )
        ax2.legend(loc='upper right', fontsize='small')
        ax1.set_xlabel('Time (s)' if i == len(channels_to_plot) - 1 else '')
    plt.suptitle(f"EMG Analysis{title_suffix}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

# --- 3. Filter Design ---
def apply_filter(data, lowcut, highcut, fs, order=4, filter_type='bandpass'):
    if data is None: return None
    nyq = 0.5 * fs
    if filter_type == 'bandpass':
        low, high = lowcut/nyq, highcut/nyq
        if not (0 < low < high < 1): print(f"Warn: Invalid bandpass. Skip filter."); return data
        b, a = butter(order, [low, high], btype='bandpass')
    elif filter_type == 'highpass':
        low = lowcut/nyq
        if not (0 < low < 1): print(f"Warn: Invalid highpass. Skip filter."); return data
        b, a = butter(order, low, btype='highpass')
    elif filter_type == 'lowpass':
        high = highcut/nyq
        if not (0 < high < 1): print(f"Warn: Invalid lowpass. Skip filter."); return data
        b, a = butter(order, high, btype='lowpass')
    else: print(f"Unknown filter type. Return original."); return data
    return filtfilt(b, a, data, axis=0)

# --- 4. Feature Extraction ---
def mav(s): return np.mean(np.abs(s), axis=0)
def rms(s): return np.sqrt(np.mean(s**2, axis=0))
def wl(s): return np.sum(np.abs(np.diff(s, axis=0)), axis=0)
def zc(s, t=1e-5): return np.sum(np.diff(np.sign(s - t), axis=0) != 0, axis=0)
def ssc(s, t=1e-5): return np.sum(np.diff(np.sign(np.diff(s, axis=0)), axis=0) != 0, axis=0)
def mnf(s, fs): 
    f, p = welch(s, fs, nperseg=len(s), axis=0); sp = np.sum(p, axis=0)
    return np.sum(f[:,None]*p,axis=0)/np.where(sp==0,1,sp)
def mdf(s, fs): 
    f, p = welch(s, fs, nperseg=len(s), axis=0); cp = np.cumsum(p, axis=0); tp = cp[-1,:]
    mi = np.apply_along_axis(lambda x:np.searchsorted(x,x[-1]/2. if x[-1]>0 else 0),0,cp)
    mi = np.clip(mi,0,len(f)-1); mv = f[mi]; mv[tp==0]=0; return mv
def extract_features_from_window(w, fs):
    return np.concatenate([mav(w),rms(w),wl(w),zc(w),ssc(w),mnf(w,fs),mdf(w,fs)])
def segment_and_extract_features(e, s_labels, fs, wd_ms, ov_pc):
    if e is None: return None,None,None
    ws_samp=int(wd_ms/1e3*fs); ov_samp=int(ws_samp*(ov_pc/100)); step=ws_samp-ov_samp
    nsa,nch=e.shape; af,al=[],[]; fnb=['MAV','RMS','WL','ZC','SSC','MNF','MDF']
    fnf=[f'{f}_Ch{c+1}' for f in fnb for c in range(nch)] # Full feature names
    for i in range(0,nsa-ws_samp+1,step):
        w=e[i:i+ws_samp,:]; wc_idx=i+ws_samp//2; cl=s_labels[wc_idx]
        if cl==0:continue # Skip rest state
        af.append(extract_features_from_window(w,fs)); al.append(cl)
    if not af:print("Warning: No features extracted.");return np.array([]),np.array([]),[]
    return np.array(af),np.array(al),fnf

# --- 5. Machine Learning Models & 6. Feature Importance ---
def train_and_evaluate_model(Xt,yt,Xv,yv,mn,mp=None,fn=None,tn=None, all_labels_for_dl=None):
    print(f"\n--- Training: {mn} ---")
    model_instance = None
    importances = None

    if mn == 'SimpleDL':
        if all_labels_for_dl is None: # Should be passed ideally
            all_unique_labels_in_split = sorted(np.unique(np.concatenate((yt, yv))))
        else:
            all_unique_labels_in_split = sorted(np.unique(all_labels_for_dl))
        
        min_label_val = min(all_unique_labels_in_split)
        
        yt_shifted = yt - min_label_val
        yv_shifted = yv - min_label_val
        
        num_classes = int(np.max(yt_shifted) + 1 if len(yt_shifted) > 0 else (np.max(yv_shifted) + 1 if len(yv_shifted) > 0 else 0) )
        if num_classes == 0 and len(all_unique_labels_in_split) > 0: # if only one class after shift
            num_classes = int(np.max(all_unique_labels_in_split - min_label_val) + 1)


        yt_cat = to_categorical(yt_shifted, num_classes=num_classes)
        # yv_cat = to_categorical(yv_shifted, num_classes=num_classes) # Not needed for evaluation with sparse labels

        model_instance = Sequential([
            Dense(128, activation='relu', input_shape=(Xt.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        model_instance.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model Architecture:")
        model_instance.summary()
        print(f"Training DL model for {DL_EPOCHS} epochs...")
        history = model_instance.fit(Xt, yt_cat, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE, validation_split=0.1, verbose=0)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title(f'{mn} Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'{mn} Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.tight_layout(); plt.show()

        y_pred_proba = model_instance.predict(Xv)
        yp_shifted = np.argmax(y_pred_proba, axis=1)
        yp = yp_shifted + min_label_val # Convert predictions back to original label scale
    
    elif mn=='SVM': model_instance=SVC(C=1.,kernel='rbf',gamma='scale',probability=True,random_state=42)
    elif mn=='RandomForest': model_instance=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
    elif mn=='KNN': model_instance=KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
    else:raise ValueError(f"Unsupported: {mn}")

    if mn != 'SimpleDL': # For non-DL models
      if mp: model_instance.set_params(**mp)
      model_instance.fit(Xt,yt)
      yp=model_instance.predict(Xv)

    acc=accuracy_score(yv,yp);f1m=f1_score(yv,yp,average='macro',zero_division=0)
    print(f"Accuracy: {acc:.4f}, F1-score (Macro): {f1m:.4f}")
    print("Classification Report:")
    ul=sorted(list(set(yv)|set(yp))) # Unique labels in true and predicted
    rtn=[tn.get(l,f"Gesture {l}")for l in ul]if tn else[f"Gesture {l}"for l in ul]
    print(classification_report(yv,yp,labels=ul,target_names=rtn,zero_division=0))
    try:
        cm=confusion_matrix(yv,yp,labels=ul)
        disp=ConfusionMatrixDisplay(cm,display_labels=rtn)
        disp.plot(cmap=plt.cm.Blues);plt.title(f'CM - {mn}');plt.xticks(rotation=45,ha='right');plt.tight_layout();plt.show()
    except Exception as e:print(f"CM plot error: {e}")
    
    if mn=='RandomForest' and fn and hasattr(model_instance, 'feature_importances_'):
        importances=model_instance.feature_importances_;si=np.argsort(importances)[::-1]
        plt.figure(figsize=(12,max(6,len(fn)//4)));plt.title(f"Importances - {mn}")
        ntf=min(20,len(fn));plt.bar(range(ntf),importances[si][:ntf],align='center')
        plt.xticks(range(ntf),np.array(fn)[si][:ntf],rotation=90);plt.tight_layout();plt.show()
        print("\nTop 10 Features:");[print(f"{i+1}. {fn[si[i]]}: {importances[si[i]]:.4f}")for i in range(min(10,len(fn)))]
    
    return {'model':mn,'accuracy':acc,'f1_macro':f1m,'trained_model':model_instance,'importances':importances}