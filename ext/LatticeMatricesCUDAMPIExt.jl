module LatticeMatricesCUDAMPIExt

using CUDA
using MPI
import LatticeMatrices: _mpi_device_aware_available

_mpi_device_aware_available(::CUDA.CuArray) =
    MPI.Initialized() && MPI.has_cuda()

end
