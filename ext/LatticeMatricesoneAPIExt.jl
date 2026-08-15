module LatticeMatricesoneAPIExt

using oneAPI
import LatticeMatrices: _backend_device_count, _select_backend_device!

_backend_device_count(::Val{:oneapi}) = length(oneAPI.devices())

function _select_backend_device!(::Val{:oneapi}, device_ordinal::Integer)
    oneAPI.device!(device_ordinal)
    return nothing
end

end
