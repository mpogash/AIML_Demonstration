def test_grad_mean_square_error(): 
    """
    Script: 
        test_grad_mean_square_error.py

    Description: 
        test script for grad_mean_square_error.py

    Development Status: 
        Complete

    Usage: 
        python -m src.data_analysis.test.test_grad_mean_square_error

    Desired Capabilities:

    Developer Notes: 
        grad_mean_square_error requires the proper order of arguments in the function call; however, 
        for testing purposes, the variable names in the latter test cases are fed "improperly" to the 
        function. The tests are valid and operate as intended, albeit the discrepency in variable names. 

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
    from src.data_analysis.grad_mean_square_error import grad_mean_square_error
    import numpy as np

    # == MACHINE TOLERANCE THRESHOLD =======================
    mtt = 1E-15

    # == BUILD BASE ARRAYS FOR TESTING =====================
    test_array = np.arange(-5,5)
    err = np.random.uniform(2,4)
    #err = np.random.randint(2,4) 
    err_sq = err**2

    # == TESTS =============================================
    # positive integers, real < truth
    assert grad_mean_square_error(1, 3) == -2

    # positive integers, real > truth
    assert grad_mean_square_error(3, 1) == 2
    
    # negative integers and 0
    assert grad_mean_square_error(-1, 0) == -1
    
    # Test vectors
    true_vec = test_array
    real_vec = true_vec+err
    assert grad_mean_square_error(real_vec,true_vec).all() == (real_vec - true_vec).all()
    assert grad_mean_square_error(true_vec,real_vec).all() == (true_vec - real_vec).all()

    # Test matrices
    #mat_1 = [vec_1, vec_1 ,vec_1] # this creates a list, would need to call np.array around it
    true_mat = np.tile(true_vec,(3,1))
    real_mat = true_mat + err 
    assert grad_mean_square_error(real_mat,true_mat).all() == (real_mat - true_mat).all()
    assert grad_mean_square_error(true_mat,real_mat).all() == (true_mat - real_mat).all()
 
    # Test tensors
    # demonstrate unique ways to make ND array (tensor) 
    true_tens = np.tile(true_mat[np.newaxis,:,:],(2,1,1))
    real_tens = np.array([real_mat, real_mat])
    assert grad_mean_square_error(real_tens,true_tens).all() == (real_tens - true_tens).all()
    assert grad_mean_square_error(true_tens,real_tens).all() == (true_tens - real_tens).all()

    # == REPORT STATUS =====================================
    print("test_mean_square_error.py successful\n")

if __name__ == "__main__":
    test_grad_mean_square_error()