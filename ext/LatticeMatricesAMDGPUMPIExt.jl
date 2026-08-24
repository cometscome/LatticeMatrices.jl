module LatticeMatricesAMDGPUMPIExt

using AMDGPU
using MPI
import LatticeMatrices: _mpi_device_aware_available

_mpi_device_aware_available(::AMDGPU.ROCArray) =
    MPI.Initialized() && MPI.has_rocm()

end
