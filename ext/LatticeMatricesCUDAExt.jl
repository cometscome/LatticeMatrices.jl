module LatticeMatricesCUDAExt

using CUDA
import LatticeMatrices: _backend_device_count, _select_backend_device!

_backend_device_count(::Val{:cuda}) = length(CUDA.devices())

function _select_backend_device!(::Val{:cuda}, device_ordinal::Integer)
    CUDA.device!(device_ordinal - 1)
    return nothing
end

end
