import os
import ctypes

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_build_dir = os.path.join(_project_root, "build")

lib_path = os.path.join(_build_dir, "libad_engine.so")
if not os.path.exists(lib_path):
    raise ImportError(
        f"Shared library not found at {lib_path}. "
        "Please compile the C library first by running 'make' in the 'c_lib' directory."
    )

try:
    ad_lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"Error loading shared library: {e}")
    print("Ensure the library is compiled for the correct architecture (32/64-bit).")
    raise

# C Type Aliases and Function Signatures
class CValue(ctypes.Structure):
    pass

CValuePtr = ctypes.POINTER(CValue)

# TapeContext Management
ad_lib.ad_init_tape.argtypes = [ctypes.c_size_t]
ad_lib.ad_init_tape.restype = None

ad_lib.ad_destroy_tape.argtypes = []
ad_lib.ad_destroy_tape.restype = None

# Variable Creation
ad_lib.ad_create_variable.argtypes = [ctypes.c_float]
ad_lib.ad_create_variable.restype = CValuePtr

# Operations
ad_lib.ad_add.argtypes = [CValuePtr, CValuePtr]
ad_lib.ad_add.restype = CValuePtr

ad_lib.ad_mul.argtypes = [CValuePtr, CValuePtr]
ad_lib.ad_mul.restype = CValuePtr

# ad_lib.ad_sin.argtypes = [CValuePtr]
# ad_lib.ad_sin.restype = CValuePtr

# ad_lib.ad_exp.argtypes = [CValuePtr]
# ad_lib.ad_exp.restype = CValuePtr

ad_lib.ad_pow.argtypes = [CValuePtr, CValuePtr]
ad_lib.ad_pow.restype = CValuePtr

# ad_lib.ad_relu.argtypes = [CValuePtr]
# ad_lib.ad_relu.restype = CValuePtr

# Backward Pass
ad_lib.ad_backward.argtypes = [CValuePtr, ctypes.c_float]
ad_lib.ad_backward.restype = None

# Gradient/Data Retrieval
ad_lib.ad_get_gradient.argtypes = [CValuePtr]
ad_lib.ad_get_gradient.restype = ctypes.c_float

ad_lib.ad_get_data.argtypes = [CValuePtr]
ad_lib.ad_get_data.restype = ctypes.c_float

# WRAPPER
class TapeContext:
    _active_tape_count = 0

    def __init__(self, initial_capacity=1024):
        self.initial_capacity = initial_capacity
        self._is_active = False
    
    def __enter__(self):
        # Only initialize if no other tape context is active
        if TapeContext._active_tape_count == 0:
            ad_lib.ad_init_tape(self.initial_capacity)
        TapeContext._active_tape_count += 1
        self._is_active = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._is_active:
            TapeContext._active_tape_count-=1
            if TapeContext._active_tape_count == 0:
                ad_lib.ad_destroy_tape()
        return False

class Variable:
    def __init__(self, value, _c_ptr=None, _children=(), _op=''):
        if TapeContext._active_tape_count == 0:
            raise RuntimeError("Variable created outside of an active TapeContext context. Use 'with TapeContext:")
        
        if _c_ptr is not None:
            self._c_ptr = _c_ptr
        elif isinstance(value, (int, float)):
            self._c_ptr = ad_lib.ad_create_variable(float(value))
        else:
            raise TypeError("Invalid type for Variable intialization")
        
        self._children = set(_children)
        self._op = _op

    @property
    def data(self):
        if not self._c_ptr: return None
        return ad_lib.ad_get_data(self._c_ptr)
    
    @property
    def grad(self):
        if not self._c_ptr: return None
        return ad_lib.ad_get_gradient(self._c_ptr)
    
    def __add__(self, other):
        other =  other if isinstance(other, Variable) else Variable(other)
        c_res_ptr = ad_lib.ad_add(self._c_ptr, other._c_ptr)
        return Variable(0, _c_ptr=c_res_ptr, _children=(self,other), _op='+')
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        c_res_ptr = ad_lib.ad_mul(self._c_ptr, other._c_ptr)
        return Variable(0, _c_ptr=c_res_ptr, _children=(self, other), _op='*')

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        c_res_ptr = ad_lib.ad_pow(self._c_ptr, other._c_ptr)
        return Variable(0, _c_ptr=c_res_ptr, _children=(self, other), _op='**')
    
    def backward(self, seed_gradient=1.0):
        ad_lib.ad_backward(self._c_ptr, float(seed_gradient))
    
    def __repr__(self):
        try:
            return f"Variable(data={self.data:.3e}, grad={self.grad:.3e})"
        except Exception:
            return f"Variable(c_ptr={self._c_ptr} - tape likely inactive)"