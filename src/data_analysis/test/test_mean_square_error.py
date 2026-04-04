def test_mean_square_error(): 
    """
    Script: 
        test_mean_square_error.py

    Description: 
        test script for mean_square_error.py

    Development Status: 
        Complete

    Usage: 
        python -m AIML_Demonstration.src.data_analysis.test.test_mean_square_error

    Desired Capabilities:

    Revision History:
        ------------------------------------------------------------------
        |  ID  |  Author     |     Date       |       Description
        ------------------------------------------------------------------
        |  0   | M.Pogash    |  04-Apr-2026   | - Initial Drop                   
        ------------------------------------------------------------------
        
    """
    # == REPORT STATUS =====================================
    print("\ncalled test_mean_square_error.py\n")

    # == IMPORT LIBRARIES ==================================
    from src.data_analysis.mean_square_error import mean_square_error
    import numpy as np

    # == MACHINE TOLERANCE THRESHOLD =======================
    mtt = 1E-15

    # == BUILD BASE ARRAYS FOR TESTING =====================
    test_array = np.arange(-5,5)
    err = np.random.uniform(2,4)
    #err = np.random.randint(2,4) 
    err_sq = err**2

    # == TESTS =============================================
    # Test two positive integers
    assert mean_square_error(1, 3) == 4
    
    # Test a negative integer and 0
    assert mean_square_error(-1, 0) == 1
    
    # Test vectors
    vec_1 = test_array
    vec_2 = vec_1+err
    #assert mean_square_error(vec_1,vec_2) == err_sq
    # account for machine precision; could also be done with np.isclose or np.testing.assert_allclose()
    assert mean_square_error(vec_1,vec_2) - err_sq < vec_1.size*mtt+mtt
    
    # Test matrices
    #mat_1 = [vec_1, vec_1 ,vec_1] # this creates a list, would need to call np.array around it
    mat_1 = np.tile(vec_1,(3,1))
    mat_2 = mat_1 + err 
    #assert mean_square_error(mat_1,mat_2) == err_sq
    # account for machine precision
    assert mean_square_error(mat_1,mat_2) - err_sq < mat_1.size*mtt+mtt
 
    # Test tensors
    # demonstrate unique ways to make ND array (tensor) 
    tens_1 = np.tile(mat_1[np.newaxis,:,:],(2,1,1))
    tens_2 = np.array([mat_2, mat_2])
    #assert mean_square_error(tens_1,tens_2) == err_sq
    # account for machine precision
    assert mean_square_error(tens_1,tens_2) - err_sq < tens_1.size*mtt+mtt

    # == REPORT STATUS =====================================
    print("test_mean_square_error.py successful\n")

if __name__ == "__main__":
    test_mean_square_error()