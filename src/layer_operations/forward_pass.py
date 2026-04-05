def forward_pass(batch_data,layers): 
    """
    Script: 
        forward_pass.py

    Description: 
        Passes data through layers of a neural network

    Development Status: 
        Complete? 
        Testing is minimal; only functional thus far, no output verification

    Usage: 
        Inputs: 
            - batch_data (nparray): array of inputs
            - layers (nested list) formatted as: 
                - layers[n][0]: nth layer weight coefficientss
                - layers[n][1]: nth layer bias coefficients 

        Outputs: 
            - output_layer (nparray): output layer(s) 
            - hidden_layers (list of lists): hidden layer values

    Desired Capabilities:
            - automatically applies relu activation to all hidden layers
              and no activation function to the output layer... give 
              more flexibility

    Revision History:
        ------------------------------------------------------------------
        |  ID  |  Author     |     Date       |       Description
        ------------------------------------------------------------------
        |  0   | M.Pogash    |  04-Apr-2026   | - Initial Drop                   
        ------------------------------------------------------------------
        
    """

    # == IMPORT LIBRARIES ===================================
    import numpy as np
    from src.layer_operations.relu import relu

    # == FORCE KEYS TO BE NPARRAY ===========================
    if not isinstance(batch_data,np.ndarray):
        batch_data = np.array(batch_data)

    # == BUILD LAYERS =======================================
    hidden_layers = [batch_data.copy()]
    for ll in range(0,len(layers)):
        batch_data = np.matmul(batch_data,layers[ll][0]) + layers[ll][1]
        if ll < len(layers) - 1:
            batch = relu(batch_data)
        hidden_layers.append(batch_data.copy())

    # == RETURN RESULT ======================================
    return batch_data, hidden_layers


