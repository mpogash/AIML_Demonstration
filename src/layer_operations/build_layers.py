def build_layers(layer_properties): 
    """
    Script: 
        build_layers.py

    Description: 
        Returns the layers of weights and biases of a neural network 
        from their perscriped properites

    Development Status: 
        In Progress

    Usage: 
        Inputs: 
            - layer_properties (dictionary) containing: 
                - neurons_per_layer (list of integers): required
                - weight_sigma_initial (list of float): default value scales to range from -1 to 1
                - bias_sigma_initial (list of float): default value scales to range from -1 to 1

        Outputs: 
            - layers (nparray): layers of weights and biases 

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

    # == POPULATE LAYER PROPERTIES ==========================
    layer_properties["n_layers"] = len(layer_properties["neurons_per_layer"])
    if not "weight_sigma_initial" in layer_properties.keys():
        layer_properties["weight_sigma_initial"] = np.ones(layer_properties["n_layers"])
    if not "bias_sigma_initial" in layer_properties.keys():
        layer_properties["bias_sigma_initial"] = np.ones(layer_properties["n_layers"])

    # == FORCE KEYS TO BE NPARRAY ===========================
    if not isinstance(layer_properties["neurons_per_layer"],np.ndarray):
        layer_properties["neurons_per_layer"] = np.array(layer_properties["neurons_per_layer"])
    if not isinstance(layer_properties["weight_sigma_initial"],np.ndarray):
        layer_properties["weight_sigma_initial"] = np.array(layer_properties["weight_sigma_initial"])
    if not isinstance(layer_properties["bias_sigma_initial"],np.ndarray):
        layer_properties["bias_sigma_initial"] = np.array(layer_properties["bias_sigma_initial"])

    # == BUILD LAYERS =======================================
    layers = []
    for ll in range(0,layer_properties["n_layers"]-1):
       layers.append([     
            layer_properties["weight_sigma_initial"][ll]*np.random.uniform(-1,1,size=(layer_properties["neurons_per_layer"][ll],layer_properties["neurons_per_layer"][ll+1])),
            layer_properties["bias_sigma_initial"][ll]*np.random.uniform(-1,1,size=(1,layer_properties["neurons_per_layer"][ll+1])),
        ])
    # == RETURN RESULT ======================================
    return layers


    """"
    # Debugging layeer
    # == BUILD LAYERS =======================================
    layers = []
    for ll in range(0,layer_properties["n_layers"]-1):
        # debugging text
        #print(f"iteration {ll} start: layers are {layers}")
        #val1 = layer_properties["weight_sigma_initial"][ll]
        #print(f"layer_properties['weight_sigma_initial'] is {val1}")
        #val2 = np.random.uniform(-1,1,size=(layer_properties["neurons_per_layer"][ll],layer_properties["neurons_per_layer"][ll+1]))
        #print(f"numpy array is {val2}")                   
        layers.append([
            # Want this to be a layer of weights and biases 
            # that are randomly generated between values defined 
            # in the layer_properties         
            #np.random.uniform(-1*layer_properties["weight_scaling"][ll],layer_properties["weight_scaling"][ll], \
            #                  size=(layer_properties["n_layers"][ll],layer_properties["n_layers"][ll-1])),
            #np.random.uniform(-1*layer_properties["bias_scaling"][ll],layer_properties["bias_scaling"][ll], \
            #                  size=(layer_properties["n_layers"][ll],layer_properties["n_layers"][ll-1])),
            layer_properties["weight_sigma_initial"][ll]*np.random.uniform(-1,1,size=(layer_properties["neurons_per_layer"][ll],layer_properties["neurons_per_layer"][ll+1])),
            layer_properties["bias_sigma_initial"][ll]*np.random.uniform(-1,1,size=(1,layer_properties["neurons_per_layer"][ll+1])),
            #val1 = layer_properties["weight_sigma_initial"]
            #print(f"layer_properties['weight_sigma_initial'] is {val1}")
            #val2 = np.random.uniform(-1,1,size=(layer_properties["neurons_per_layer"][ll],layer_properties["neurons_per_layer"][ll+1]))
            #print(f"numpy array is {val2}")
            #layer_properties["bias_sigma_initial"][ll]*np.random.uniform(-1,1,size=(1,layer_properties["neurons_per_layer"][ll+1])),
        ])
        #print(f"iteration {ll} end: layers are {layers}")
        """