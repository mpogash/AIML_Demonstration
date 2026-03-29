"""
Script: 
    linear_regression_neural_network.py

Description: 
    This script programs a linear regression neural network (NN) using first principles. No AIML libraries
    are used for the developed NN; the model is built from scratch using basic Python and NumPy.
    The script generates synthetic data, trains the linear regression model, and evaluates its performance.
    Performance is evaluated using scikit-learn's LinearRegression model as a benchmark for comparison as well
    as a tensorflow NN. The script also includes visualization of the synthetic data and the results of the 
    various models; figures are save to the specified directory.

    The script follows the work of "Neural Network From Scratch In Python" by Dataquest on YouTube.
    
Development Status: 
    Incomplete. 
    Scipt currently performs the following:
        - Generates synthetic data
        - Trains a linear regression model using scikit-learn's LinearRegression class
        - Visualizes the synthetic data and the predictions of the linear regression model
    The following capabilities are desired but not yet implemented:
        - Train a linear regression model using a custom-built neural network (NN) from scratch 
        - Train a linear regression model using a NN built with tensorflow
        - Evaluate the performance of the custom-built NN with respect to the tensorflow NN against the
          scikit-learn model

Usage: 
    1. Modify the user-defined parameters in the "DEFINE USER INPUTS" section as desired.
    2. Change to the main directory of the project in the terminal: 
           cd /home/mike/GitHub/AIML_Demonstration/).
    3. Run the script:
           python -m scripts.linear_regression_neural_network
    4. Observe output figures created in the specified directory.

Desired Capabilities:
    1. Data Visualization: 
        a. Training Progress: 
            - figure with 2 subplots: 
                - subplot 1: Loss curve 
                - subplot 2: CDF of errors for all samples, a unique CDF is created for each epoch
                             and overlaid on the same figure

        b. Model Performance Comparison:
            - figure with 1 subplot:
                - for the existing figure showing the x and y data, convert markers to a scatter density 
                  plot if the number of samples is very large. 
                - Add a textbox to the figure specifying the fit details for each model, such as weights, 
                  bias, R^2 score, etc.) and the training details (e.g., n_epochs, learning_rate, etc.)                    

    2. Activation Function Variation:
        a. Implement the ability to select different activation functions (e.g., ReLU, Sigmoid, Tanh)
           for the hidden layers of the NN and compare their performance on the linear regression
    
    3. Function Geenralization:
        a. Some of the lines of code here should be converted to funcitons, including the 
           synthetic data generation. 

Revision History:
    ------------------------------------------------------------------
    |  ID  |  Author     |     Date       |       Description
    ------------------------------------------------------------------
    |  0   | M.Pogash    |  29-Mar-2026   | - Initial Drop                   
    ------------------------------------------------------------------
    
"""

# ========== IMPORT LIBRARIES ==============
from more_itertools import tail
from datetime import datetime
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Custom imports
from src.data_generation.gen_synthetic_data_linear import gen_synthetic_data_linear
#from GitHub.AIML_Demonstration.src.data_generation.gen_synthetic_data_linear import gen_synthetic_data_linear

# ========= DEFINE USER INPUTS ============
synthetic_data_properties = {
    "n_samples": 5E2,
    "feature_weights": [4],
    "bias": 4,
    "absolute_noise_scalar": 1.0,
    "relative_noise_scalar_fraction": 0.1,
    "x_data_range": (0, 10)
}
neural_network_properties = {
    "learning_rate": 0.01,
    "n_epochs": 1000,
    "early_stopping_threshold": 0.001
}

configuration_details = {
    "figure_generation_switch": True,
    "figure_directory": f"/home/mike/GitHub/AIML_Demonstration/figures/runs{datetime.now().date()}"
}
# ========= END USER INPUT DEFINITION =====
 
# ============= INITATION ==============
# Set random seed for reproducibility
np.random.seed(0)

# Ensure user-defined values are appropriate.
if type(synthetic_data_properties["n_samples"]) != int:
    synthetic_data_properties["n_samples"] = int(synthetic_data_properties["n_samples"])
    print(f"n_samples was a float, but has been converted to an integer: {synthetic_data_properties['n_samples']}")

# Append n_features into synthetic data properties
synthetic_data_properties["n_features"] = len(synthetic_data_properties["feature_weights"])

# Create names for feature array
synthetic_data_properties["feature_names"] = [f"Feature_{i}" for i in range(synthetic_data_properties["n_features"])]

# Create directory for saving figures if it doesn't exist
figures_directory = configuration_details["figure_directory"]
if configuration_details["figure_generation_switch"]:
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

# ============= GENERATE SYNTHETIC DATA ==============
# Generate synthetic data for linear regression
synthetic_data = gen_synthetic_data_linear(synthetic_data_properties)

# ============= TRAIN LINEAR REGRESSION MODEL ==============
# Prepare data for training
lr_truth_model = LinearRegression()
lr_truth_model.fit(synthetic_data[synthetic_data_properties["feature_names"]], synthetic_data["y_data"])
lr_truth_model_predictions = lr_truth_model.predict(synthetic_data[synthetic_data_properties["feature_names"]])
lr_truth_model_weights = lr_truth_model.coef_
lr_truth_model_bias = lr_truth_model.intercept_
del lr_truth_model
x_min = synthetic_data[synthetic_data_properties["feature_names"]].min().values[0]
x_max = synthetic_data[synthetic_data_properties["feature_names"]].max().values[0]
lr_truth_model_fit_x = (x_min, x_max)
del x_min, x_max
lr_truth_model_fit_y = tuple(lr_truth_model_fit_x * lr_truth_model_weights + lr_truth_model_bias)

# =============== VISAULIZE SYNTHETIC DATA =================
if configuration_details["figure_generation_switch"]:
    ax = synthetic_data.plot.scatter(synthetic_data_properties["feature_names"][0], "y_data", alpha=0.5, color='blue', label="Synthetic Data")
    plt.plot(lr_truth_model_fit_x,lr_truth_model_fit_y, color="red", label="Truth Model Predictions")
    plt.legend()
    figure_savepath = f"{configuration_details["figure_directory"]}{os.sep}synthetic_data_scatter_plot.png"
    plt.savefig(figure_savepath)
    print(f"figure saved:  {figure_savepath}")

"""
# Troubleshooting Snipbits

print(f"types of synthetic_data_properties samples: \n"
      f"n_samples: {type(synthetic_data_properties["n_samples"])}\n"
      f"n_features: {type(synthetic_data_properties["n_features"])}\n")

print(f"x_data is of size {x_data.shape} ") #and has a tail of \n {tail(x_data)}')     

print(f"{lr_truth_model_fit_y.tail}")

#print(f"{synthetic_data[synthetic_data_properties["feature_names"]].tail}")

#print(f"{lr_truth_model_fit_x}")

"""