def test_forward_pass(): 
    """
    Script: 
        test_forward_pass.py

    Description: 
        test script for forward_pass.py

    Development Status: 
        In-Progress
        Tests only functional. Need 

    Usage: 
        python -m src.layer_operations.test.test_forward_pass

    Desired Capabilities:

    Developer Notes: 
        Would be nice to make the assert lines more general to prevent 
        re-thinking concepts if stronger tests are needed

    Revision History:
        ------------------------------------------------------------------
        |  ID  |  Author     |     Date       |       Description
        ------------------------------------------------------------------
        |  0   | M.Pogash    |  04-Apr-2026   | - Initial Drop                   
        ------------------------------------------------------------------
        
    """
    # == REPORT STATUS =====================================
    print("\ncalled test_forward_pass.py\n")

    # == IMPORT LIBRARIES ==================================
    from src.layer_operations.forward_pass import forward_pass
    from src.layer_operations.build_layers import build_layers
    import numpy as np

    # == BUILD LAYERS AND DATA FOR TESTING ==================
    batch_data = np.arange(-10,10)
    n_neurons_layer_0 = len(batch_data)
    
    # test with all keys defined
    layer_properties_1 = {
        "neurons_per_layer": [n_neurons_layer_0, n_neurons_layer_0*2,n_neurons_layer_0//2, 1],
        "weight_sigma_initial": np.random.uniform(-3,3,size=4),
        "bias_sigma_initial": np.random.uniform(-2,2,size=4),
    }
    # test with only required keys
    layer_properties_2 = {
        "neurons_per_layer": [n_neurons_layer_0, 1],
    }

    # == TESTS =============================================
    layers_1 = build_layers(layer_properties_1)
    batch_data_1, hidden_layers_1 = forward_pass(batch_data,layers_1)

    layers_2 = build_layers(layer_properties_2)
    batch_data_2, hidden_layers_2 = forward_pass(batch_data,layers_2)

      # == REPORT STATUS =====================================
    print("testforward_pass.py successful\n")
    print("testforward_pass.py only consists of a functional test. No output verificaiton is performed\n")

if __name__ == "__main__":
    test_forward_pass()

"""
# DEBUG TEXT

    print(f"layer_1 was build with {layer_properties_1}")    
    for key,value in layer_properties_1.items():
       print(f"key: {key}, value: {value}")
    print(f"layers_1 is {layers_1}")

    
    print(f"layer_2 was build with {layer_properties_2}")    
    for key,value in layer_properties_2.items():
       print(f"key: {key}, value: {value}")
    print(f"layers_2 is {layers_2}")
   
    print(f"layers_2, layer_0 weight shape is {layers_2[0][0].shape} and bias shape is {layers_2[0][1].shape}")
    print(f"layers_2, layer_1 weight shape is {layers_2[1][0].shape} and bias shape is {layers_2[1][1].shape}")

"""