def test_gen_synthetic_data_linear(): 
    """
    Script: 
        test_gen_synthetic_data_linear.py

    Description: 
        test script for gen_synthetic_data_linear.py

    Development Status: 
        Complete

    Usage: 
        python -m AIML_Demonstration.src.data_generation.test.test_gen_synthetic_data_linear

    Desired Capabilities:
        improve robustness of testing
    
    Revision History:
        ------------------------------------------------------------------
        |  ID  |  Author     |     Date       |       Description
        ------------------------------------------------------------------
        |  0   | M.Pogash    |  04-Apr-2026   | - Initial Drop                   
        ------------------------------------------------------------------
        
    """
    # == REPORT STATUS =====================================
    print("\ncalled test_gen_synthetic_data_linear.py\n")

    # == IMPORT LIBRARIES ==================================
    from src.data_generation.gen_synthetic_data_linear import gen_synthetic_data_linear
    import numpy as np

    # == BUILD BASE DICTIONARIES FOR TESTING =====================
    # test with 1 variable and appropiate data type definitions
    dict_1 = {
        "n_samples": 5E3,
        "feature_weights": [4],
        "bias": 4.2,
        "absolute_noise_scalar": 1.0,
        "relative_noise_scalar_fraction": 0.1,
        "x_data_range": (0, 10)
    }

    # == TESTS =============================================
    syn_data = gen_synthetic_data_linear(dict_1) 
    print(f"size of syn_data y_data is {syn_data["y_data"].size} ")
    assert syn_data["y_data"].size == dict_1["n_samples"]
    # check that the appropiate keys were added to the dictionaries
    assert "feature_names" in dict_1.keys()
    assert "n_features" in dict_1.keys()

    # == REPORT STATUS =====================================
    print("test_gen_synthetic_data_linear.py successful\n")

if __name__ == "__main__":
    test_gen_synthetic_data_linear()