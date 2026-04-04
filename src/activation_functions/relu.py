def relu(data): 
    """
    Script: 
        relu.py

    Description: 
        Returns the data after relu activation function is applied

    Development Status: 
        Complete

    Usage: 
        Inputs: 
            - data (ndarray (preferred), array, matrix, tensor): The true values.
        Outputs: 
            - data_activated (ndarray): data transformated by relu function 

    Desired Capabilities:

    Revision History:
        ------------------------------------------------------------------
        |  ID  |  Author     |     Date       |       Description
        ------------------------------------------------------------------
        |  0   | M.Pogash    |  04-Apr-2026   | - Initial Drop                   
        ------------------------------------------------------------------
        
    """
    import numpy as np
    if not isinstance(data, (np.ndarray)):
        data = np.array(data)

    data_activated = np.maximum(0 , data)
    return data_activated