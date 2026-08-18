module LatticeMatricesCUDAExt

using CUDA
import LatticeMatrices: _backend_device_count, _prepare_mpi_host_buffer,
                        _select_backend_device!

_backend_device_count(::Val{:cuda}) = length(CUDA.devices())

function _select_backend_device!(::Val{:cuda}, device_ordinal::Integer)
    CUDA.device!(device_ordinal - 1)
    return nothing
end

# Register the existing Julia Array with CUDA instead of changing its public
# type. Host-staged MPI then avoids CUDA's additional pageable-memory bounce.
_prepare_mpi_host_buffer(::CUDA.CuArray, buffer::Array) = CUDA.pin(buffer)

end
