# Beamforming Design for Large-Scale Antenna Arrays Using Deep Learning - Sensitivity Analysis

This repository contains the PyTorch implementation and analysis code for the project "Beamforming Design for Large-Scale Antenna Arrays Using Deep Learning," which investigates the sensitivity of the Beamforming Neural Network (BFNN) proposed by Lin and Zhu [1] to various training and channel parameters.

The core BFNN architecture aims to directly generate analog beamformers from estimated (imperfect) Channel State Information (CSI) to maximize Spectral Efficiency (SE) in millimeter wave (mmWave) MISO systems. This project focuses on evaluating how SE performance is affected by:
*   **Training Parameters:** Activation functions, learning rates, training epochs.
*   **Channel Parameters:** CSI quality (PNR) and the assumed number of channel paths (Lest) during estimation.

The original work [1] and associated MATLAB code for dataset generation can be found [here](https://github.com/TianLin0509/BF-design-with-DL).

## Prerequisites

*   **Python:** Version 3.10.16 (as used in the report) or compatible.
*   **PyTorch:** Version 2.6.0 (as used in the report) or compatible.
*   **Other Python Libraries:** See `requirements.txt`.
*   **MATLAB:** Required *only* if you need to re-generate the channel datasets from scratch using the scripts in `make_dataset/`. The necessary training/testing data (`.mat` files) may already be included in the `train_set/` directory.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lexuanhoang120/Beam-forming-with-deep-learning.git
    cd Beam-forming-with-deep-learning
    ```

2.  **Set up a Python virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

*   The channel data generation process follows the methodology described in [1] and uses the hierarchical codebook-based estimator from [2].
*   The MATLAB scripts for generating the `.mat` dataset files are located in the `make_dataset/` directory (specifically `gen_samples.m`).
*   **Pre-generated data:** This repository likely includes the necessary pre-generated training and testing data (`H_est.mat`, `H_perfect.mat`, etc.) within the `train_set/` directory. You should be able to run the Python training and evaluation scripts without needing MATLAB if this data is present.


## Reproducing the Results

Follow these steps in order to reproduce the experiments described in the report. Ensure the required dataset files are present in `train_set/`.

### Step 1: Train Models

Run the training scripts from your terminal. Trained model checkpoints will be saved in the corresponding `models_*` directories.

*   **Parameter-based Scenario (Section 6.1 - Training Parameter Analysis):**
    This script will iterate through the different activation functions, learning rates, and epoch settings defined within it.
    ```bash
    python train_parameter.py
    ```

*   **Channel-based Scenario (Section 6.2 - Channel Parameter Analysis):**
    This script trains models using the chosen hyperparameters from Section 6.1.5, varying PNR and Lest as defined within the script or configuration.
    ```bash
    python train_channel.py
    ```
    *(Note: This assumes `train_channel.py` uses the fixed best parameters found previously. Check the script if needed.)*

### Step 2: Evaluate Trained Models

After training is complete, run the evaluation scripts. These scripts load the saved models and evaluate them on the test dataset, saving the results to `.csv` files.

*   **Parameter-based Scenario (Section 6.1):**
    ```bash
    python test_parameter.py
    ```
    (This will generate/update `evaluation_results_parameter.csv`)

*   **Channel-based Scenario (Section 6.2):**
    ```bash
    python test_channel.py
    ```
    (This will generate/update `evaluation_results_channel.csv`)

### Step 3: Analyze and Visualize Results

Open and run the Jupyter Notebook to load the results from the `.csv` files and generate the plots and analyses presented in the report (e.g., Tables 2, 3, 4 and Figures 4, 5).

*   **Open Jupyter Notebook:**
    ```bash
    jupyter notebook result.ipynb
    ```
    Run the cells within the notebook to visualize the results.

## Citation

If you use this code or the findings from the original paper, please cite:

[1] T. Lin and Y. Zhu, “Beamforming design for large-scale antenna arrays using deep learning,” *IEEE Wireless Communications Letters*, vol. 9, no. 1, pp. 60–64, Jan. 2020.

[2] A. Alkhateeb, O. E. Ayach, G. Leus, and R. W. Heath, “Channel estimation and hybrid precoding for millimeter wave cellular systems,” *IEEE Journal of Selected Topics in Signal Processing*, vol. 8, pp. 831–846, Oct. 2014.

## Author

*   LE XUAN HOANG - 24114545
