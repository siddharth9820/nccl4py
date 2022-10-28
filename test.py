import torch
torch.ops.load_library('build/libnccl4py.so')
from utils import init_dist

if __name__ == "__main__":
    init_dist()

    torch.ops.nccl4py.setup_communicator_and_stream(3)

    rank = torch.ops.nccl4py.get_world_rank()
    size = torch.ops.nccl4py.get_world_size()
    nccl_stream = torch.cuda.ExternalStream(torch.ops.nccl4py.get_stream())

    x = torch.ones(2,3, device='cuda').half()
    torch.ops.nccl4py.all_reduce_fp16(x)
    torch.cuda.synchronize()
    print(x)
