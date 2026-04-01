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

    See the  ./figures/linear_regression_neural_network/ directory for a demonstration and parametric 
              evaluation of number of epochs and learning rate         

Development Status: 
    Partially Complete. 
    Scipt currently performs the following:
        - Generates synthetic data
        - Trains a linear regression model using scikit-learn's LinearRegression class
        - Trains a linear regression model using a NN built with tensorflow
        - Visualizes the synthetic data and the predictions of the linear regression model as well as reisduals   
    The following capabilities are desired but not yet implemented:
        - Train a linear regression model using a custom-built neural network (NN) from scratch 

Usage: 
    1. Modify the user-defined parameters in the "DEFINE USER INPUTS" section as desired.
    2. Change to the main directory of the project in the terminal: 
           cd /home/mike/GitHub/AIML_Demonstration/).
    3. Run the script:
           python -m scripts.linear_regression_neural_network
    4. Observe output figures created in the specified directory.

Tips: 
    1. When setting the learning rate, start with a small value (e.g., 0.001) and adjust as 
       needed based on the convergence of the model. If model is converging poorly, reduce 
       the learning rate by an order of magnitude. This will occur if the solution space
       does not have a smooth gradient. If the solution space is pristine, increase the 
       learning rate by an order of magnitude to speed up convergence.

Desired Capabilities:
    0. Configuration File Driven Architecture:
        a. The script should be driven by a configuration file

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
        a. Some of the lines of code here should be converted to funcitons

Revision History:
    ------------------------------------------------------------------
    |  ID  |  Author     |     Date       |       Description
    ------------------------------------------------------------------
    |  0   | M.Pogash    |  29-Mar-2026   | - Initial Drop                   
    ------------------------------------------------------------------
    
"""

# == IMPORT LIBRARIES ==================================================================
from more_itertools import tail
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -- Suppress TensorFlow logging messages-- 
# 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
# Optional: To specifically silence the oneDNN message
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# -- End of TensorFlow logging suppression --
import tensorflow as tf
from tensorflow.keras import layers, models

# Custom imports
from src.data_generation.gen_synthetic_data_linear import gen_synthetic_data_linear

# == DEFINE USER INPUTS =====================================================================
# Modify the user-defined parameters in this section as desired.
# Future iterations will use a configuration file

synthetic_data_properties = {
    "n_samples": 5E3,
    "feature_weights": [4],
    "bias": 4.2,
    "absolute_noise_scalar": 1.0,
    "relative_noise_scalar_fraction": 0.1,
    "x_data_range": (0, 10)
}

configuration_details = {
    "figure_generation_switch": True,
    "figure_directory": f"/home/mike/GitHub/AIML_Demonstration/figures/linear_regession_neural_network/runs{datetime.now().date()}"
}

tf_properties = {
    "n_epochs": 100,
    "test_size": 0.6,
    "batch_size": 32,
    "random_state": 42,
    "output_units": 1,
    "num_neurons": 1,
    "learning_rate": 0.01,
    "loss_metric": "mse",
    "report_metrics": ["mae"],
}   

# not currently implement. 
neural_network_properties = {
    "learning_rate": 0.001,
    "n_epochs": 1000,
    "early_stopping_threshold": 0.001,
    "activation_function": "linear"
}

# == END USER INPUT DEFINITION ==============================================================

# WARNING. General users should not modify the code below. 
 
# == INITATION ==============================================================================
# Set random seed for reproducibility
np.random.seed(0)

# Ensure user-defined values are appropriate.
if type(synthetic_data_properties["n_samples"]) != int:
    synthetic_data_properties["n_samples"] = int(synthetic_data_properties["n_samples"])
    print(f"n_samples was a float, but has been converted to an integer: {synthetic_data_properties['n_samples']}")

if type(synthetic_data_properties["bias"]) == int:
    synthetic_data_properties["bias"] = synthetic_data_properties["bias"]+1E-16
    print(f"bias was an integer, but has been converted to a float: {synthetic_data_properties['bias']}")   

# Append n_features into synthetic data properties
synthetic_data_properties["n_features"] = len(synthetic_data_properties["feature_weights"])

# Create names for feature array
synthetic_data_properties["feature_names"] = [f"Feature_{i}" for i in range(synthetic_data_properties["n_features"])]

# Create directory for saving figures if it doesn't exist
figures_directory = configuration_details["figure_directory"]
if configuration_details["figure_generation_switch"]:
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

# == GENERATE SYNTHETIC DATA ====================================================================
# Generate synthetic data for linear regression
synthetic_data = gen_synthetic_data_linear(synthetic_data_properties)

# == TRAIN LINEAR REGRESSION MODEL ==============================================================
# Prepare data for training
lr_truth_model = LinearRegression()
lr_truth_model.fit(synthetic_data[synthetic_data_properties["feature_names"]], synthetic_data["y_data"])
lr_truth_model_weights = lr_truth_model.coef_
lr_truth_model_bias = lr_truth_model.intercept_
# only keep neceessary info 
del lr_truth_model

# == TRAIN A NN USING TENSORFLOW =================================================================
x_train, x_test, y_train, y_test = train_test_split(synthetic_data[synthetic_data_properties["feature_names"]].values, \
                                                    synthetic_data["y_data"].values, test_size=tf_properties["test_size"], \
                                                    random_state=tf_properties["random_state"])

# normalize the feature data using standard scaling (zero mean, unit variance)
feature_normalizer = StandardScaler()
x_train_normalized = feature_normalizer.fit_transform(x_train)
x_test_normalized = feature_normalizer.transform(x_test)   
del x_train

tf_model = models.Sequential()
# note: 1 neurons, with a weight for each feature. The bias terms is included 
#       in the Dense layer by default. The input shape is defined as [N,] and not
#       [N,1] because we want to specify a 1D vector of features. 
tf_model.add(layers.Input(shape=(synthetic_data_properties["n_features"],)))
tf_model.add(layers.Dense(tf_properties["num_neurons"], activation="linear"))
tf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tf_properties["learning_rate"]), \
                 loss=tf_properties["loss_metric"], metrics=tf_properties["report_metrics"]) 
tf_model.fit(x_train_normalized, y_train, epochs=tf_properties["n_epochs"], \
             validation_data=(x_test_normalized, y_test), batch_size=tf_properties["batch_size"], \
             verbose=0)
tf_model_weights, tf_model_bias = tf_model.layers[0].get_weights()
# undo the normalization of weights for reporting and evaluation
tf_normalizer_mu = feature_normalizer.mean_
tf_normalizer_sigma = feature_normalizer.scale_
tf_model_weights_unnormalized = tf_model_weights / tf_normalizer_sigma 
tf_model_bias_unnormalized = tf_model_bias - np.sum(tf_model_weights * tf_normalizer_mu / tf_normalizer_sigma) 

# == EVALUATE TENSORFLOW NN PERFORMANCE =========================================================
#lr_truth_model_predictions = lr_truth_model.predict(x_test) #synthetic_data[synthetic_data_properties["feature_names"]])
defined_model_predictions = synthetic_data_properties["feature_weights"] * x_test + synthetic_data_properties["bias"]   
lr_truth_model_predictions = lr_truth_model_weights * x_test + lr_truth_model_bias
tf_model_predictions = tf_model.predict(x_test_normalized)

# compute residuals 
lr_truth_model_residuals = defined_model_predictions - lr_truth_model_predictions
tf_model_residuals = defined_model_predictions - tf_model_predictions

# == EVALUATE PERFORMANCE OF MODELS ==============================================================
print(f"Defined Model Weights: {synthetic_data_properties['feature_weights']}, Defined Model Bias: {synthetic_data_properties['bias']}")
print(f"Standrd Linear Regression Weights: {lr_truth_model_weights}, Truth Model Bias: {lr_truth_model_bias}")
print(f"TensorFlow Linear Regression Weights: {tf_model_weights_unnormalized}, TensorFlow Model Bias: {tf_model_bias_unnormalized}")

# == VISUALIZATION ===================================================================
if configuration_details["figure_generation_switch"]:
    # figure options
    c_syn_marker = [0.5, 0.5, 0.5]
    c_syn = [0, 0, 0]
    c_lr = [1, 0, 0]
    c_tf = [0, .5, 1]
    fs_labels = 14
    fs_text = 12
    fs_legends = 12
    lw = 3
    ms = 2.5

    # set global font size for figures
    plt.rcParams.update({'font.size': fs_labels})

    # define path for output figure
    learning_rate_str = str.replace(str(tf_properties["learning_rate"]), ".", "p") # replace decimal point with 'p' for filename    
    figure_filename = f"results_tf_nEpochs_{tf_properties['n_epochs']}_LearningRate_{learning_rate_str}.png"
    figure_savepath = f"{configuration_details["figure_directory"]}{os.sep}{figure_filename}"

    #  visualize specific data 
    x_min = synthetic_data[synthetic_data_properties["feature_names"]].min() #.values[0]
    x_max = synthetic_data[synthetic_data_properties["feature_names"]].max() #.values[0]
    dataset_x_bounds = np.array([x_min, x_max])
    del x_min, x_max
    dataset_x_bounds_scaled = feature_normalizer.transform(dataset_x_bounds)
    defined_model_fit_y = synthetic_data_properties["feature_weights"] * dataset_x_bounds + synthetic_data_properties["bias"]
    lr_truth_model_fit_y = lr_truth_model_weights * dataset_x_bounds + lr_truth_model_bias
    tf_model_fit_y = tf_model_weights * dataset_x_bounds_scaled + tf_model_bias

    # create figure         
    fig, fig_ax = plt.subplots(1, 2, figsize=(18, 6))
    
    # Subplot 1: Scatter plot of synthetic data and model predictions
    fig_ax[0].scatter(synthetic_data[synthetic_data_properties["feature_names"][0]], synthetic_data["y_data"], s = ms, alpha=0.3, color=c_syn_marker, label="Synthetic Data")
    fig_ax[0].plot(dataset_x_bounds, defined_model_fit_y, color=c_syn, linestyle ='-', linewidth=lw, label="Defined Model")
    fig_ax[0].plot(dataset_x_bounds, lr_truth_model_fit_y, color=c_lr, linestyle ='--', linewidth=lw, label="Standard Linear Regression")
    fig_ax[0].plot(dataset_x_bounds, tf_model_fit_y, color=c_tf, linestyle =':', linewidth=lw*1.25, label="TensorFlow Linear Regression")
    fig_ax[0].set_title("Synthetic Data and Fits")
    fig_ax[0].set_xlabel(synthetic_data_properties["feature_names"][0])
    fig_ax[0].set_ylabel("y_data")
    fig_ax[0].legend(loc="upper left", fontsize=fs_legends)  
    label_str = f"Defined Model: Weights={synthetic_data_properties['feature_weights'][0]:.2f}, Bias={synthetic_data_properties['bias']:.2f}\n" \
        f"Linear Regression: Weights={lr_truth_model_weights[0]:.2f}, Bias={lr_truth_model_bias:.2f}\n" \
        f"TensorFlow Model: Weights*={tf_model_weights_unnormalized[0][0]:.2f}, Bias*={tf_model_bias_unnormalized[0]:.2f}\n\n" \
        f"TensorFlow Training: n_epochs={tf_properties['n_epochs']}, learning_rate={tf_properties['learning_rate']}"
    fig_ax[0].text(0.98, 0.02, label_str, transform=fig_ax[0].transAxes, fontsize=fs_text, verticalalignment='bottom', \
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0)) 
    # Subplot 2: Residuals
    n_bins = synthetic_data_properties["n_samples"] // 20 # set number of bins based on number of samples
    fig_ax[1].hist(lr_truth_model_residuals, bins=n_bins, density=True, alpha=0.5, color=c_lr, label="Standard Linear Regression")
    fig_ax[1].hist(tf_model_residuals, bins=n_bins, density=True, alpha=0.5, color=c_tf, label="TensorFlow Linear Regression")
    fig_ax[1].set_title("Model Residuals")
    fig_ax[1].set_xlabel("Residual Value")
    fig_ax[1].set_ylabel("Counts")
    fig_ax[1].set_yscale('log')
    fig_ax[1].legend(fontsize=fs_legends)  
     
    # save figure  
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