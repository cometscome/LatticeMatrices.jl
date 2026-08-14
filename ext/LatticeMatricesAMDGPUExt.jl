module LatticeMatricesAMDGPUExt

using AMDGPU
import LatticeMatrices: _backend_device_count, _select_backend_device!

_backend_device_count(::Val{:amdgpu}) = length(AMDGPU.devices())

function _select_backend_device!(::Val{:amdgpu}, device_ordinal::Integer)
    AMDGPU.device_id!(device_ordinal)
    return nothing
end

end
