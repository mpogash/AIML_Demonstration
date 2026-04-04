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
        make testing more robust

    Revision History:
        ------------------------------------------------------------------
        |  ID  |  Author     |     Date       |       Description
        ------------------------------------------------------------------
        |  0   | M.Pogash    |  04-Apr-2026   | - Initial Drop                   
        ------------------------------------------------------------------
        
    """
    print("\ncalled test_mean_square_error.py\n")

    from AIML_Demonstration.src.data_analysis.mean_square_error import mean_square_error

    # Test two positive integers
    assert mean_square_error(1, 3) == 4
    # Test a negative integer and 0
    assert mean_square_error(-1, 0) == 1
    # Test vectors
    # Test matrices
    # Test tensors
    print("\ntest_mean_square_error.py successful\n")

if __name__ == "__main__":
    test_mean_square_error()