def test_relu(): 
    """
    Script: 
        test_relu.py

    Description: 
        test script for relu.py

    Development Status: 
        Complete

    Usage: 
        python -m src.layer_operations.test.test_relu

    Desired Capabilities:

    Revision History:
        ------------------------------------------------------------------
        |  ID  |  Author     |     Date       |       Description
        ------------------------------------------------------------------
        |  0   | M.Pogash    |  04-Apr-2026   | - Initial Drop                   
        ------------------------------------------------------------------
        
    """
    # == REPORT STATUS =====================================
    print("\ncalled test_relu.py\n")

    # == IMPORT LIBRARIES ==================================
    from src.layer_operations.relu import relu
    import numpy as np

    # == BUILD BASE ARRAYS FOR TESTING =====================
    # test with floats
    test_array_1 = np.random.uniform(-10,10,150)
    # test including 0
    test_array_2 = np.arange(-5,5)

    # == TESTS =============================================
    assert np.min(relu(test_array_1)) >= 0 
    assert np.min(relu(test_array_2)) >= 0 

    # == REPORT STATUS =====================================
    print("test_relu.py successful\n")

if __name__ == "__main__":
    test_relu()