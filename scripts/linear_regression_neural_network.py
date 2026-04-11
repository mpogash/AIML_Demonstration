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
    - Partially complete
            - main goal accomplished, but, fine tuning could still be performed 

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
        a. Add a logarithmic rounding funciton for figures
        b. Add R^2 to text box of figures                   

    2. Activation Function Variation:
        a. Implement the ability to select different activation functions (e.g., ReLU, Sigmoid, Tanh)
           for the hidden layers of the NN and compare their performance on the linear regression

    3. Add visualization for training loss of custom neural network

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
import sys

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
from src.layer_operations.build_layers import build_layers
from src.layer_operations.forward_pass import forward_pass
from src.layer_operations.backward_pass import backward_pass
from src.data_analysis.grad_mean_square_error import grad_mean_square_error
from src.data_analysis.mean_square_error import mean_square_error

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
    "n_epochs": 10,
    "learning_rate": 0.1,
    "test_size": 0.6,
    "batch_size": 32,
    "random_state": 42,
    "output_units": 1,
    "num_neurons": 1,
    "loss_metric": "mse",
    "report_metrics": ["mse"], # remove? Not used in current iteration
}   

# not currently implement. 
cust_nn_properties = {
    "learning_rate": 0.01,
    "neurons_per_layer": [1, 1],
    "weight_sigma_initial": [1, 1],
    "bias_sigma_initial": [1, 1],
    "n_epochs": 50,
    "batch_size": 32,
    "early_stopping_threshold": 0.01,
    "activation_function": "linear"
    # "n_layers": [1, 1, 1], autopopulated
}

# == END USER INPUT DEFINITION ==============================================================

# WARNING. General users should not modify the code below. 
 
# == INITATION ==============================================================================
# Set random seed for reproducibility
np.random.seed(0)

# Create directory for saving figures if it doesn't exist
figures_directory = configuration_details["figure_directory"]
if configuration_details["figure_generation_switch"]:
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

# == GENERATE SYNTHETIC DATA ====================================================================
# Generate synthetic data for linear regression, synthetic data is a pandas dataframe
synthetic_data = gen_synthetic_data_linear(synthetic_data_properties)

print(f"size of synthetic_data['x_data'] is {synthetic_data["Feature_0"].shape}")
print(f"size of synthetic_data['y_data'] is {synthetic_data["y_data"].shape}")

# == SPLIT DATA INTO TEST AND TRAIN DATA =======================================================
x_train, x_test, y_train, y_test = train_test_split(synthetic_data[synthetic_data_properties["feature_names"]].values, \
                                                    synthetic_data["y_data"].values, test_size=tf_properties["test_size"], \
                                                    random_state=tf_properties["random_state"])

# ensure y data is matrix, not vector
if y_train.ndim == 1:
    y_train = y_train.reshape(-1,1)

if y_test.ndim == 1:
    y_test = y_test.reshape(-1,1)

# normalize the feature data using standard scaling (zero mean, unit variance)
feature_normalizer = StandardScaler()
x_train_normalized = feature_normalizer.fit_transform(x_train)
x_test_normalized = feature_normalizer.transform(x_test)   
del x_train

# == TRAIN LINEAR REGRESSION MODEL ==============================================================
# Prepare data for training
lr_truth_model = LinearRegression()
lr_truth_model.fit(synthetic_data[synthetic_data_properties["feature_names"]], synthetic_data["y_data"])
lr_truth_model_weights = lr_truth_model.coef_
lr_truth_model_bias = lr_truth_model.intercept_
# only keep neceessary info 
del lr_truth_model

# == TRAIN A NN USING CUSTOM-BUILT NEURAL NETWORK FROM SCRATCH ===================================
cust_nn_mse_hist = []
layers_c_nn = build_layers(cust_nn_properties)
print(f"epoch loop not started")
print(f"layers are {layers_c_nn}")
for ee in range(cust_nn_properties["n_epochs"]):
    for bb in range(0,y_train.shape[0],cust_nn_properties["batch_size"]):
        x_train_bb = x_train_normalized[bb:(bb+cust_nn_properties["batch_size"])]
        y_train_bb = y_train[bb:(bb+cust_nn_properties["batch_size"])]

        y_inferred_bb,hidden_layer = forward_pass(x_train_bb,layers_c_nn)
        grad_bb = grad_mean_square_error(y_inferred_bb,y_train_bb)
        layers_c_nn = backward_pass(layers_c_nn,hidden_layer,grad_bb,cust_nn_properties)
        #print(f"epoch is {ee}, batch index start is {bb} and data sampling start is {bb} and stop is {(bb+cust_nn_properties["batch_size"])}\n")
        #print(f"layers_c_nn are {layers_c_nn}")
        #print(f"mean of train loss is {np.mean(loss_bb)}\n")
        #print(f"epoch loss is {epoch_loss}\n")
    y_inferred_val,_= forward_pass(x_test_normalized,layers_c_nn)
    mse = mean_square_error(y_inferred_val, y_test)
    cust_nn_mse_hist.append(mse)
    #print(f"y inferrred values are  {y_inferred_val[:5]}")
    #print(f"y inferrred values are  {y_inferred_val[-5:]}")
    #print(f"epoch is {ee}, Train MSE is {mse}")
    #print(f"layers are {layers}")

cust_nn_model_weights = layers_c_nn[0][0]
cust_nn_model_bias = layers_c_nn[0][1]
# undo the normalization of weights for reporting and evaluation
tf_normalizer_mu = feature_normalizer.mean_
tf_normalizer_sigma = feature_normalizer.scale_
cust_nn_model_weights_unnormalized = cust_nn_model_weights / tf_normalizer_sigma 
cust_nn_model_bias_unnormalized = cust_nn_model_bias - np.sum(cust_nn_model_weights * tf_normalizer_mu / tf_normalizer_sigma) 

# == TRAIN A NN USING TENSORFLOW =================================================================
tf_model = models.Sequential()
# note: 1 neurons, with a weight for each feature. The bias terms is included 
#       in the Dense layer by default. The input shape is defined as [N,] and not
#       [N,1] because we want to specify a 1D vector of features. 
tf_model.add(layers.Input(shape=(synthetic_data_properties["n_features"],)))
#tf_model.add(layers.Input(shape=(synthetic_data_properties["n_features"])))
tf_model.add(layers.Dense(tf_properties["num_neurons"], activation="linear"))
tf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tf_properties["learning_rate"]), \
                 loss=tf_properties["loss_metric"], metrics=tf_properties["report_metrics"]) 
tf_model_history = tf_model.fit(x_train_normalized, y_train, epochs=tf_properties["n_epochs"], \
             validation_data=(x_test_normalized, y_test), batch_size=tf_properties["batch_size"], \
             verbose=0)
tf_model_weights, tf_model_bias = tf_model.layers[0].get_weights()
# undo the normalization of weights for reporting and evaluation
tf_normalizer_mu = feature_normalizer.mean_
tf_normalizer_sigma = feature_normalizer.scale_
tf_model_weights_unnormalized = tf_model_weights / tf_normalizer_sigma 
tf_model_bias_unnormalized = tf_model_bias - np.sum(tf_model_weights * tf_normalizer_mu / tf_normalizer_sigma) 

# == EVALUATE PERFORMANCE OF MODELS =========================================================
#lr_truth_model_predictions = lr_truth_model.predict(x_test) #synthetic_data[synthetic_data_properties["feature_names"]])
defined_model_predictions = synthetic_data_properties["feature_weights"] * x_test + synthetic_data_properties["bias"]   
lr_truth_model_predictions = lr_truth_model_weights * x_test + lr_truth_model_bias
tf_model_predictions = tf_model.predict(x_test_normalized)

# compute residuals 
lr_truth_model_residuals = lr_truth_model_predictions - defined_model_predictions
tf_model_residuals = tf_model_predictions - defined_model_predictions
cust_nn_model_residuals = y_inferred_val - defined_model_predictions

# == EVALUATE PERFORMANCE OF MODELS ==============================================================
#print(f"Defined Model Weights: {synthetic_data_properties['feature_weights']}, Defined Model Bias: {synthetic_data_properties['bias']}")
#print(f"Standrd Linear Regression Weights: {lr_truth_model_weights}, Truth Model Bias: {lr_truth_model_bias}")
#print(f"TensorFlow Linear Regression Weights: {tf_model_weights_unnormalized}, TensorFlow Model Bias: {tf_model_bias_unnormalized}")

# == VISUALIZATION ===================================================================
if configuration_details["figure_generation_switch"]:
    # figure options
    fig_sz_x = 18
    fig_sz_y = 5
    c_syn_marker = [0.5, 0.5, 0.5]
    c_syn = [0, 0, 0]
    c_lr = [1, 0, 0]
    c_tf = [0, .5, 1]
    c_c_nn = [.7, 0, 1] 
    ms_train_test_loss = 4
    c_train_loss_tf = c_tf #[0, 0, 0]
    marker_train_loss_tf = "s"
    c_test_loss_tf = [val * 0.75 for val in c_train_loss_tf] #[0.45, 0.45, 0.45]
    marker_test_loss_tf = "x"
    c_train_loss_c_nn = c_c_nn #[0, 0, 0]
    marker_train_loss_c_nn = "s"
    c_test_loss_c_nn = [val * 0.75 for val in c_train_loss_c_nn] #[0.45, 0.45, 0.45]
    marker_test_loss_c_nn = "x"
    fs_labels = 12
    fs_text = 8
    fs_legends = 10
    lw = 3
    ms = 2.5
    lw_grid = 0.5
    alpha_grid = 0.5

    # set global font size for figures
    plt.rcParams.update({'font.size': fs_labels})

    # define path for output figure
    c_nn_learning_rate_str = str.replace(str(cust_nn_properties["learning_rate"]), ".", "p") # replace decimal point with 'p' for filename   
    tf_learning_rate_str = str.replace(str(tf_properties["learning_rate"]), ".", "p") # replace decimal point with 'p' for filename    
    figure_filename = f"results_custom_NN_nEpochs_{cust_nn_properties['n_epochs']}_LearningRate_{c_nn_learning_rate_str}_tf_nEpochs_{tf_properties['n_epochs']}_LearningRate_{tf_learning_rate_str}.png"
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
    cust_nn_fit_y = layers_c_nn[0][0] * dataset_x_bounds_scaled + layers_c_nn[0][1]

    # create figure         
    fig, fig_ax = plt.subplots(1, 3, figsize=(fig_sz_x, fig_sz_y))
    
    # Subplot 1: Scatter plot of synthetic data and model predictions
    fig_ax[0].scatter(synthetic_data[synthetic_data_properties["feature_names"][0]], synthetic_data["y_data"], s = ms, alpha=0.3, color=c_syn_marker, label="Synthetic Data")
    fig_ax[0].plot(dataset_x_bounds, defined_model_fit_y, color=c_syn, linestyle ='-', linewidth=lw, label="Defined Model")
    fig_ax[0].plot(dataset_x_bounds, lr_truth_model_fit_y, color=c_lr, linestyle ='--', linewidth=lw, label="Standard Linear Regression")
    fig_ax[0].plot(dataset_x_bounds, tf_model_fit_y, color=c_tf, linestyle =':', linewidth=lw*1.25, label="TensorFlow Linear Regression")
    fig_ax[0].plot(dataset_x_bounds, cust_nn_fit_y, color=c_c_nn, linestyle = '-.', linewidth=lw*0.75, label="Custom NN Linear Regression")
    fig_ax[0].set_title("Synthetic Data and Fits")
    fig_ax[0].set_xlabel(synthetic_data_properties["feature_names"][0])
    fig_ax[0].set_ylabel("y_data")
    fig_ax[0].legend(loc="upper left", fontsize=fs_legends)  
    label_str = f"Defined Model: Weights={synthetic_data_properties['feature_weights'][0]:.2f}, Bias={synthetic_data_properties['bias']:.2f}\n" \
        f"Linear Regression: Weights={lr_truth_model_weights[0]:.2f}, Bias={lr_truth_model_bias:.2f}\n" \
        f"TensorFlow Model: Weights*={tf_model_weights_unnormalized[0][0]:.2f}, Bias*={tf_model_bias_unnormalized[0]:.2f}\n" \
        f"Custom NN Model: Weights*={cust_nn_model_weights_unnormalized[0][0]:.2f}, Bias*={cust_nn_model_bias_unnormalized[0][0].item():.2f}\n\n" \
        f"TensorFlow Training: n_epochs={tf_properties['n_epochs']}, learning_rate={tf_properties['learning_rate']}\n" \
        f"Custom NN raining: n_epochs={cust_nn_properties['n_epochs']}, learning_rate={cust_nn_properties['learning_rate']}"

        #f"Custom NN Model: Weights*={layers_c_nn[0][0].flatten()[0]:.2f}, Bias*={layers_c_nn[0][1].item():.2f}\n\n" \
    
    fig_ax[0].text(0.98, 0.02, label_str, transform=fig_ax[0].transAxes, fontsize=fs_text, verticalalignment='bottom', \
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0)) 
    
    # Subplot 2: Residuals
    n_bins = synthetic_data_properties["n_samples"] // 20 # set number of bins based on number of samples
    fig_ax[1].hist(lr_truth_model_residuals, bins=n_bins, density=True, alpha=0.5, color=c_lr, label="Standard Linear Regression")
    fig_ax[1].hist(tf_model_residuals, bins=n_bins, density=True, alpha=0.5, color=c_tf, label="TensorFlow Linear Regression")
    fig_ax[1].hist(cust_nn_model_residuals, bins=n_bins, density=True, alpha=0.5, color=c_c_nn, label="Custom NN Linear Regression")
    fig_ax[1].set_title("Model Residuals")
    fig_ax[1].set_xlabel("Residual Value")
    fig_ax[1].set_ylabel("Counts")
    fig_ax[1].set_yscale('log')
    fig_ax[1].legend(fontsize=fs_legends)  
    # logarithmic rounding for y-axis limits
    y_min, y_max = fig_ax[1].get_ylim() 
    y_ax_min = 10**(np.floor(np.log10(y_min)))
    y_ax_max = 10**(np.ceil(np.log10(y_max)))
    fig_ax[1].set_ylim(y_ax_min, y_ax_max)  
    fig_ax[1].grid(which='both', linestyle='--', linewidth=lw_grid, alpha=alpha_grid)

    # Subplot 3: Training Progress Figure
    y_train_loss = tf_model_history.history["loss"]
    y_val_loss = tf_model_history.history["val_loss"]
    if tf_properties["loss_metric"] == "mse":
        y_train_loss = np.sqrt(y_train_loss)
        y_val_loss = np.sqrt(y_val_loss)
        y_axis_label = "RMSE"
    elif tf_properties["loss_metric"] == "mae":
        y_axis_label = "MAE"
    else:
        y_axis_label = tf_properties["loss_metric"] 

    # add capability to track training loss
    #fig_ax[2].plot(y_train_loss, marker=marker_train_loss, color=c_train_loss_tf, linestyle='none', ms = ms_train_test_loss, linewidth=lw, label="Custom NN Training Loss")
    fig_ax[2].plot(cust_nn_mse_hist, marker=marker_test_loss_c_nn, color=c_test_loss_c_nn, linestyle='none', ms = ms_train_test_loss, linewidth=lw, label="Custom NN Validation Loss")
    fig_ax[2].plot(y_train_loss, marker=marker_train_loss_tf, color=c_train_loss_tf, linestyle='none', ms = ms_train_test_loss, linewidth=lw, label="TensorFlow Training Loss")
    fig_ax[2].plot(y_val_loss, marker=marker_test_loss_tf, color=c_test_loss_tf, linestyle='none', ms = ms_train_test_loss, linewidth=lw, label="TensorFlow Validation Loss")
    fig_ax[2].set_title("Model Training Progress")
    fig_ax[2].set_xlabel("Epoch")
    fig_ax[2].set_ylabel(y_axis_label)
    fig_ax[2].legend(fontsize=fs_legends)       
    fig_ax[2].set_yscale('log')
    fig_ax[2].grid(which='both', linestyle='--', linewidth=lw_grid, alpha=alpha_grid)
    # logarithmic rounding for y-axis limits
    y_min, y_max = fig_ax[2].get_ylim() 
    y_ax_min = 10**(np.floor(np.log10(y_min)))
    y_ax_max = 10**(np.ceil(np.log10(y_max)))
    print(f"y_min is {y_min}, y_max is {y_max}, y_ax_min is {y_ax_min}, y_ax_max is {y_ax_max}")
    fig_ax[2].set_ylim(y_ax_min, y_ax_max) 
    
    # save figure  
    plt.savefig(figure_savepath)
    print(f"figure saved:  {figure_savepath}")



"""
# Troubleshooting Snipbits
print(f"y_min is {y_min}, y_max is {y_max}, y_ax_min is {y_ax_min}, y_ax_max is {y_ax_max}")

print(f"types of synthetic_data_properties samples: \n"
      f"n_samples: {type(synthetic_data_properties["n_samples"])}\n"
      f"n_features: {type(synthetic_data_properties["n_features"])}\n")

print(f"x_data is of size {x_data.shape} ") #and has a tail of \n {tail(x_data)}')     

print(f"{lr_truth_model_fit_y.tail}")

#print(f"{synthetic_data[synthetic_data_properties["feature_names"]].tail}")

#print(f"{lr_truth_model_fit_x}")

"""