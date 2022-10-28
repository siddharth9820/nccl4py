### NCCL4Py: Python bindings that are an interface to NCCL

I am developing this because PyTorch's interface to NCCL while being user friendly is not flexible enough for low-level usecases

## Compiling

```
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ../

```
