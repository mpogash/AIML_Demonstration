def test_build_layers(): 
    """
    Script: 
        test_build_layers.py

    Description: 
        test script for build_layers.py

    Development Status: 
        Complete

    Usage: 
        python -m src.layer_operations.test.test_build_layers

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
    print("\ncalled test_build_layers.py\n")

    # == IMPORT LIBRARIES ==================================
    from src.layer_operations.build_layers import build_layers
    import numpy as np

    # == BUILD BASE ARRAYS FOR TESTING =====================
    # test with all keys defined
    layer_properties_1 = {
        "neurons_per_layer": [10, 20, 20, 10, 1],
        "weight_sigma_initial": np.random.uniform(-3,3,size=5),
        "bias_sigma_initial": np.random.uniform(-2,2,size=5),
    }
    # test with only required keys
    layer_properties_2 = {
        "neurons_per_layer": [10, 5, 1],
    }

    # == TESTS =============================================
    layers_1 = build_layers(layer_properties_1)
    layers_2 = build_layers(layer_properties_2)

    assert layers_1[0][0].shape == (layer_properties_1["neurons_per_layer"][0],layer_properties_1["neurons_per_layer"][1])
    assert layers_1[0][1].shape == (1,layer_properties_1["neurons_per_layer"][1])
    assert layers_1[1][0].shape == (layer_properties_1["neurons_per_layer"][1],layer_properties_1["neurons_per_layer"][2])
    assert layers_1[1][1].shape == (1,layer_properties_1["neurons_per_layer"][2])
    assert layers_1[2][0].shape == (layer_properties_1["neurons_per_layer"][2],layer_properties_1["neurons_per_layer"][3])
    assert layers_1[2][1].shape == (1,layer_properties_1["neurons_per_layer"][3])   

    assert layers_2[0][0].shape == (layer_properties_2["neurons_per_layer"][0],layer_properties_2["neurons_per_layer"][1])
    assert layers_2[0][1].shape == (1,layer_properties_2["neurons_per_layer"][1])
    assert layers_2[1][0].shape == (layer_properties_2["neurons_per_layer"][1],layer_properties_2["neurons_per_layer"][2])
    assert layers_2[1][1].shape == (1,layer_properties_2["neurons_per_layer"][2])

    # == REPORT STATUS =====================================
    print("test_build_layers.py successful\n")

if __name__ == "__main__":
    test_build_layers()

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