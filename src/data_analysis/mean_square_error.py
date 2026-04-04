def mean_square_error(true_data, actual_data): 
    """
    Script: 
        mse.py

    Description: 
        Returns the mean squared error between true and actual_data.

    Development Status: 
        Complete

    Usage: 
        Inputs: 
            - true_data (ndarray (preferred), array, matrix, tensor): The true values.
            - actual_data (ndarray (preferred), array, matrix, tensor): The actual values.
        Outputs: 
            - mse (float): The mean squared error between the true and actual data.

    Desired Capabilities:

    Revision History:
        ------------------------------------------------------------------
        |  ID  |  Author     |     Date       |       Description
        ------------------------------------------------------------------
        |  0   | M.Pogash    |  04-Apr-2026   | - Initial Drop                   
        ------------------------------------------------------------------
        
    """

    # == IMPORT LIBRARIES ===================================
    import numpy as np

    # == CONVERT TO NPARRAYS ================================
    if not isinstance(true_data, (np.ndarray)):
        true_data = np.array(true_data)

    if not isinstance(actual_data, (np.ndarray)):
        actual_data = np.array(actual_data)

    # == COMPUTE MSE ========================================
    mse = np.mean((true_data - actual_data) ** 2)
    
    # == RETURN THE MSE =====================================
    return mse