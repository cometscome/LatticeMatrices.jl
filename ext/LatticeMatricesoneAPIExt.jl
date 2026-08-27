module LatticeMatricesoneAPIExt

using oneAPI
import LatticeMatrices: _backend_device_count, _select_backend_device!,
                        _uses_full_halo_buffers

_backend_device_count(::Val{:oneapi}) = length(oneAPI.devices())

# oneArray does not reliably support the compact derived views used by the
# optimized halo exchange. Keep oneAPI on its existing host-staged transport,
# but pack full preallocated face buffers as the portable fallback.
_uses_full_halo_buffers(::oneAPI.oneArray) = true

function _select_backend_device!(::Val{:oneapi}, device_ordinal::Integer)
    oneAPI.device!(device_ordinal)
    return nothing
end

end
