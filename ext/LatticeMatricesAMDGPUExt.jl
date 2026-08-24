module LatticeMatricesAMDGPUExt

using AMDGPU
import LatticeMatrices: _backend_device_count, _select_backend_device!,
                        _mpi_device_buffer_supported, _mpi_device_kind

_backend_device_count(::Val{:amdgpu}) = length(AMDGPU.devices())

function _select_backend_device!(::Val{:amdgpu}, device_ordinal::Integer)
    AMDGPU.device_id!(device_ordinal)
    return nothing
end

_mpi_device_buffer_supported(::AMDGPU.ROCArray) = true
_mpi_device_kind(::AMDGPU.ROCArray) = :rocm

end
