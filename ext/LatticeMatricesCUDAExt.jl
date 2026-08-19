module LatticeMatricesCUDAExt

using CUDA
using MPI
import LatticeMatrices: _backend_device_count, _prepare_mpi_host_buffer,
                        _select_backend_device!, _mpi_device_aware_available,
                        _mpi_device_buffer_supported, _mpi_device_kind

_backend_device_count(::Val{:cuda}) = length(CUDA.devices())

function _select_backend_device!(::Val{:cuda}, device_ordinal::Integer)
    CUDA.device!(device_ordinal - 1)
    return nothing
end

# Register the existing Julia Array with CUDA instead of changing its public
# type. Host-staged MPI then avoids CUDA's additional pageable-memory bounce.
_prepare_mpi_host_buffer(::CUDA.CuArray, buffer::Array) = CUDA.pin(buffer)

_mpi_device_buffer_supported(::CUDA.CuArray) = true
_mpi_device_aware_available(::CUDA.CuArray) = MPI.Initialized() && MPI.has_cuda()
_mpi_device_kind(::CUDA.CuArray) = :cuda

end
