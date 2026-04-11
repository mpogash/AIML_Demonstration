def mean_square_error(real_data, true_data): 
    """
    Script: 
        mean_square_error.py

    Description: 
        Returns the mean squared error between true_data and real_data.

    Development Status: 
        Complete

    Usage: 
        Inputs: 
            - real_data (ndarray (preferred), array, matrix, tensor): The real/actual/measured/observed values.
            - true_data (ndarray (preferred), array, matrix, tensor): The true/predicted values.
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

    if not isinstance(real_data, (np.ndarray)):
        real_data = np.array(real_data)

    # == COMPUTE MSE ========================================
    mse = np.mean((real_data - true_data) ** 2)
    
    # == RETURN THE MSE =====================================
    return mse