import torch
import numpy as np

# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)
# print(x_data)
# print(x_data.shape)

# # numpy 배열로부터 생성 numpy 버전에 따라 안됨
# np_array = np.array(data).reshape
# x_np = torch.from_numpy(np_array)
# print(x_np)
#
# x_ones = torch.ones_like(x_data)
# print(f'Ones Tensor >> \n', x_ones)
#
# x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(x_rand)

# tensor = torch.rand(3,4)
# if torch.backends.mps.is_available():
#     tensor = tensor.to('mps')
#     print(f"Device tensor is stored on: {tensor.device}")

tensor = torch.ones(4, 4)
tensor[:, 1] = 3
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

print(tensor*tensor)
