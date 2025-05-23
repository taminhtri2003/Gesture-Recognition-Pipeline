# Gesture-Recognition-Pipeline

# EMG Signal Processing and Classification Pipeline for NinaPro DB5

This project provides a Python-based pipeline for processing and classifying Electromyography (EMG) signals from the NinaPro Database 5 (DB5). It includes steps for data loading, visualization, filtering, feature extraction (time-domain and frequency-domain), and machine learning model training and evaluation.

## Features

* **Data Loading**: Loads EMG data, stimulus labels, and sampling frequency from NinaPro DB5 `.mat` files.
* **Signal Visualization**:
    * Plots raw EMG signals with corresponding stimulus labels.
    * Plots filtered EMG signals with stimulus labels and an optional Root Mean Square (RMS) envelope for smoothing visualization.
* **Signal Filtering**: Customizable Butterworth filter (bandpass, highpass, lowpass).
* **Feature Extraction**:
    * **Time-Domain Features**: Mean Absolute Value (MAV), Root Mean Square (RMS), Waveform Length (WL), Zero Crossings (ZC), Slope Sign Changes (SSC).
    * **Frequency-Domain Features**: Mean Frequency (MNF), Median Frequency (MDF).
    * Features are extracted from sliding windows with configurable duration and overlap.
* **Machine Learning**:
    * Trains and evaluates multiple classifiers: Support Vector Machine (SVM), Random Forest, K-Nearest Neighbors (KNN).
    * Data is scaled before training.
* **Performance Evaluation**:
    * Calculates accuracy and F1-score (macro).
    * Displays a detailed classification report.
    * Plots confusion matrices for each model.
* **Feature Importance**: For Random Forest, visualizes the importance of each feature.
* **Model Comparison**: Presents a summary table and bar chart comparing the performance of the different models.

## Prerequisites

* Python 3.7+
* The following Python libraries (see `requirements.txt` for specific versions):
    * `numpy`
    * `scipy`
    * `matplotlib`
    * `scikit-learn`
    * `pandas`

## Setup

1.  **Clone the repository (or download the script):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NinaPro DB5 Data:**
    * Obtain the NinaPro DB5 dataset from the [official NinaPro website](http://ninapro.hevs.ch/).
    * Place the relevant `.mat` files (e.g., `S5_E2_A1.mat`) in your project directory or a known location.

## Usage

1.  **Configure the script:**
    Open the Python script (`your_script_name.py`) and modify the following configuration parameters at the top of the file:
    * `MAT_FILE_PATH`: Set this to the path of your NinaPro DB5 `.mat` file.
        ```python
        MAT_FILE_PATH = "path/to/your/S_E_A.mat"
        ```
    * `FILTER_TYPE`, `LOWCUT`, `HIGHCUT`, `FILTER_ORDER`: Adjust filter settings if needed.
    * `WINDOW_DURATION_MS`, `WINDOW_OVERLAP_PERCENT`: Modify windowing parameters for feature extraction.
    * `CHANNELS_FOR_VISUALIZATION`: Select which EMG channels to display in the detailed plots.
    * `RMS_WINDOW_FOR_PLOT_MS`: Set the RMS window for visualization.

2.  **Run the script:**
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file).

The script will then:
* Load the specified `.mat` file.
* Display plots of the raw and filtered EMG signals.
* Extract features.
* Train the machine learning models.
* Print evaluation metrics and display confusion matrices and feature importance plots.
* Show a final comparison of model performances.

## File Structure

.├── your_script_name.py   # Main Python script for the pipeline├── README.md             # This file├── requirements.txt      # Python dependencies├── .gitignore            # Files to be ignored by Git└── data/                 # Optional: Directory to store your .mat files└── S5_E2_A1.mat      # Example .mat file (not included in repo)
## Customization

* **Filters**: Modify `LOWCUT`, `HIGHCUT`, `FILTER_ORDER`, and `FILTER_TYPE` in the script.
* **Features**: Add or remove feature extraction functions in the `extract_features_from_window` function. Remember to update `feature_names_base` if you do.
* **Models**: Add new classifiers to the `train_and_evaluate_model` function and the `models_to_evaluate` list in the main pipeline.
* **Windowing**: Adjust `WINDOW_DURATION_MS` and `WINDOW_OVERLAP_PERCENT` for feature extraction.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or find bugs.

## License

(Optional: Add a license here, e.g., MIT License. If you do, create a `LICENSE` file as well.)
