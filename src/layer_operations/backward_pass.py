def backward_pass(layers,hidden_layers,grad,train_properties): 
    """
    Script: 
        backward_pass.py

    Description: 
        Passes data through layers of a neural network

    Development Status: 
        In-Progress

    Usage: 
        Inputs: 
            - layers (nested list): array of inputs
                - layers[n][0]: nth layer weight coefficientss
                - layers[n][1]: nth layer bias coefficients 
            - hidden_layers (nested list) formatted as: 
                - hidden_layers[n][0]: nth hidden layer weight coefficientss
                - hidden_layers[n][1]: nth hidden layer bias coefficients 
            - grad
            - train_properties (dictionary) containing: 
                - "learning_rate" (float): learing rate
                

        Outputs: 
            - output_layer (nparray): output layer(s) 
            - hidden_layers (list of lists): hidden layer values

    Desired Capabilities:
            - assumes relu activation was applied to all hidden layers
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

    # == ITERATE BACKWARD THROUGH LAYERS =====================
    #print(f"shape of layers is  {len(layers)} , {len(layers[0])}")
    for ll in range(len(layers)-1,-1,-1):
        if ll != len(layers)-1:
            # assume relu activation function
            grad = np.multiply(grad, np.heaviside(hidden_layers[ll+1],0))
        
        weight_grad = hidden_layers[ll].T @ grad
        bias_grad = np.mean(grad, axis=0)
        #print(f"shape of hidden_layer[layer_i].T is {len(hidden_layers[ll].T)} , {len(hidden_layers[ll].T[0])}")
        #print(f"shape of grad is {len(grad)} , {len(grad[0])}")
        #print(f"shape of weight grad is {len(weight_grad)} , {len(weight_grad[0])}")
        #print(f"shape of layers is  {len(layers)} , {len(layers[0])}")
        layers[ll][0] -= weight_grad * train_properties["learning_rate"]
        layers[ll][1] -= bias_grad * train_properties["learning_rate"]
        grad = grad @ layers[ll][0].T

    # == RETURN RESULT ======================================
    return layers


