def gen_synthetic_data_linear(synthetic_data_properties): 
    """
Script: 
    gen_synthetic_data_linear.py

Description: 
    Generates linear synthetic data based on a dictionary of user-defined properties.
        
Development Status: 
    Complete

Usage: 
    Inputs: 
        - synthetic_data_properties (dict): A dictionary containing the properties for the synthetic data generation.
            Required keys include:
                - n_samples (int): The number of samples to generate.
                - feature_weights (list): A list of weights for each feature in the linear equation.
                - bias (float): The bias term to be added to the linear equation.
                - absolute_noise_scalar (float): A scalar to control the magnitude of absolute noise added to the data.
                - relative_noise_scalar_fraction (float): A scalar fraction to control the magnitude of relative noise 
                  added to the data.
                - x_data_range (tuple): A tuple specifying the range (min, max) for the uniform distribution from which 
                  the feature data will be drawn.
    Outputs: 
        - synthetic_data (pandas DataFrame): A DataFrame containing the generated synthetic data 
          with feature columns and a target variable column named "y_data".  

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
    import numpy as np
    import pandas as pd

    x_data = np.random.uniform(synthetic_data_properties["x_data_range"][0],
                               synthetic_data_properties["x_data_range"][1],
                               (synthetic_data_properties["n_samples"], synthetic_data_properties["n_features"]))
    y_data = synthetic_data_properties["feature_weights"] * x_data + synthetic_data_properties["bias"]
    y_data = y_data + synthetic_data_properties["absolute_noise_scalar"] * np.random.randn(synthetic_data_properties["n_samples"], 1) + \
             synthetic_data_properties["relative_noise_scalar_fraction"] * y_data * np.random.randn(synthetic_data_properties["n_samples"], 1)
    # Convert to pandas DataFrame 
    synthetic_data = pd.DataFrame(x_data, columns=synthetic_data_properties["feature_names"])
    synthetic_data["y_data"] = y_data
    return synthetic_data