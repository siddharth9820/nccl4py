#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>
#include "cuda.h"
#include "nccl.h"
#include "cuda_runtime.h"
#include "mpi.h"


// retrived from: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


ncclComm_t nccl_comm;
cudaStream_t stream;

void print_tensor(const torch::Tensor& x)
{
	std::cout << x << std::endl;
}

int64_t get_world_rank()
{	
	int world_rank;
	NCCLCHECK(ncclCommUserRank(nccl_comm, &world_rank));
	return (int64_t)world_rank;
}


int64_t get_world_size()
{	
	int world_size;
	NCCLCHECK(ncclCommCount(nccl_comm, &world_size));
	return (int64_t)world_size;
}

int64_t get_device()
{	
	int device;
	cudaGetDevice(&device);
	return (int64_t)device;
}

void setup_communicator_and_stream(int64_t G_intra)
{
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	ncclUniqueId nccl_id, nccl_ids[world_size];
	size_t id_size = sizeof(ncclUniqueId);
	NCCLCHECK(ncclGetUniqueId(&nccl_id));
	MPI_Allgather(&nccl_id, id_size, MPI_UINT8_T,
                &nccl_ids[0], id_size, MPI_UINT8_T, MPI_COMM_WORLD);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	nccl_id = nccl_ids[rank / G_intra];

	NCCLCHECK(ncclCommInitRank(&nccl_comm, G_intra, nccl_id, rank % G_intra));
	int priority_high, priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low , &priority_high ) ;

	CUDACHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority_high ));
}

int64_t get_stream()
{
	return (int64_t) stream;
}


void all_reduce_fp16(const torch::Tensor& x)
{
	NCCLCHECK(ncclAllReduce(x.data_ptr<at::Half>(), x.data_ptr<at::Half>(), x.numel(), ncclHalf, ncclSum, nccl_comm, stream));
}

void all_reduce_fp32(const torch::Tensor& x)
{
	NCCLCHECK(ncclAllReduce(x.data_ptr<float>(), x.data_ptr<float>(), x.numel(), ncclFloat, ncclSum, nccl_comm, stream));
}

void all_gather_fp16(const torch::Tensor& x, torch::Tensor& y)
{
	NCCLCHECK(ncclAllGather(x.data_ptr<at::Half>(), y.data_ptr<at::Half>(), x.numel(), ncclHalf, nccl_comm, stream));
}

void all_gather_fp32(const torch::Tensor& x, torch::Tensor& y)
{
	NCCLCHECK(ncclAllGather(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), ncclFloat, nccl_comm, stream));
}

TORCH_LIBRARY(nccl4py, m) {
  m.def("print_tensor", print_tensor);
  m.def("get_world_rank", get_world_rank);
  m.def("get_world_size", get_world_size);
  m.def("get_device", get_device);
  m.def("get_stream", get_stream);
  m.def("setup_communicator_and_stream", setup_communicator_and_stream);
  m.def("all_reduce_fp16", all_reduce_fp16);
  m.def("all_reduce_fp32", all_reduce_fp32);
  m.def("all_gather_fp16", all_gather_fp16);
  m.def("all_gather_fp32", all_gather_fp32);
}

