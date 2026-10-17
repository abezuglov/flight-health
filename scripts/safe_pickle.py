"""Load the NGAFID pickle, allowing only the numpy types it was scanned to contain."""
import pickle, numpy as np

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}

class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"blocked: {module}.{name}") from None

def load(path):
    with open(path, "rb") as f:
        return _Restricted(f).load()
