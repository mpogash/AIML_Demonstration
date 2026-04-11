import numpy as np
y_sz_data = 28
batch_sz  = 5

data_array = np.arange(0,y_sz_data)
#data_array = np.random.uniform(0,1,y_sz_data)
#print(f"{data_array}")

y_sz_c = data_array.shape[0]
idx_b_i = list(range(0,y_sz_c,batch_sz))
print(f"idx_b_i is {idx_b_i}")

idx_b_f = [val - 1 for val in idx_b_i]
print(f"idx_b_f is {idx_b_f}")

idx_b_f.pop(0)
print(f"idx_b_f is {idx_b_f}")

idx_b_f.append(y_sz_c+1)
print(f"idx_b_f is {idx_b_f}")


print(f"\nidx_b_i is {idx_b_i}")
print(f"idx_b_f is {idx_b_f}\n")

#print(f"{data_array[idx_b_i[0]:idx_b_f[0]]}")

n_batches = len(idx_b_i)
for ii in range(0,n_batches):
    print(f"{data_array[idx_b_i[ii]:idx_b_f[ii]+1]}")




#print(f"type of idx_b_f data is {type(idx_b_f)}")
#print(f"idx_b_i[0] is {idx_b_i[0]} and type if {type(idx_b_i[0])}") 
#print(f"idx_b_f[0] is {idx_b_f[0]} and type if {type(idx_b_f[0])}") 

#print(f"1 is {data_array[[idx_b_i[0]:[idx_b_f[0]]}")