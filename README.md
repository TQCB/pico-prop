# PicoPonder: A Reverse-Mode Automatic Differentiation Engine

PicoPonder is a lightweight automatic differentiation (AD) engine implemented in C, designed for performing reverse-mode AD (backpropagation). It provides a C API that can be easily interfaced from Python using `ctypes`, making it suitable for building custom neural network components or for other gradient-based optimization tasks where performance of the AD core is important.

This project was inspired by the need for a understandable and extensible AD system, using a C core for potential benefits, and to explore C-Python interoperability.

## Features

*   **Reverse-Mode Automatic Differentiation:** Efficiently computes gradients of a scalar output with respect to many inputs.
*   **Dynamic Computation Graph:** Builds a tape of operations during the forward pass.
*   **C Core:** Core AD logic implemented in C for performance.
*   **Python Interface:** Easy-to-use Python wrapper (`Variable` class) using `ctypes` for seamless integration.
*   **Basic Arithmetic:**
    *   Addition (`+`)
    *   Multiplication (`*`)
*   **Tape Management:** Python context manager (`TapeContext`) for managing the lifecycle of the computation graph.

## How it Works

PicoProp implements reverse-mode automatic differentiation using a "tape-based" approach:

1.  **Forward Pass:**
    *   When operations are performed on `Variable` objects (which wrap C pointers), the C core records each operation and its operands on a "tape" (a list of `t_node` structs).
    *   Each `t_node` struct stores its data, its gradient (initialized to 0), its parent(s) in the computation graph, and a function pointer to its specific backward operation.

2.  **Backward Pass:**
    *   Starting from the final output `Variable`, its gradient is seeded (usually to 1.0).
    *   The C core then traverses the tape in reverse chronological order.
    *   For each `t_node` on the tape, its stored backward operation function is called. This function calculates the gradients of its inputs based on its own gradient, using the chain rule, and accumulates these gradients into the parent `t_node` structs.

## Project Structure

```
pico_prop/
├── src/
│ ├── ad_engine.c # C implementation of the AD engine
│ └── ad_engine.h # Header file for the C engine
├── include/ # Public headers
│ └── ad_engine.h
├── python/
│ └── pico_prop.py # Python wrapper using ctypes (contains Variable, TapeContext)
├── build/ # Directory for compiled shared library (e.g., libad_engine.so)
├── Makefile # Makefile for building the C shared library
└── README.md # This file!!!!!
```

## Prerequisites

*   A C compiler (e.g., GCC, Clang)
*   `make` (or adjust build commands accordingly)
*   Python 3.x

## Building the C Core

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/TQCB/pico-prop
    cd pico_prop
    ```

2.  **Compile the shared library:**
    Navigate to the project root directory (where the `Makefile` is located) and run:
    ```bash
    make
    ```
    This will compile `src/ad_engine.c` into a shared library (currently only `build/libad_engine.so` on Linux).

    Alternatively, you can compile manually (example for Linux):
    ```bash
    mkdir -p build
    gcc -shared -o build/libad_engine.so -fPIC src/ad_engine.c -Iinclude -lm
    ```
    Adjust the `-Iinclude` path if your public headers are elsewhere. The `-lm` links the math library.

## Usage (Python)

The primary way to use PicoPonder is through its Python interface.

```python
from pico_prop import Variable, TapeContext

if __name__ == "__main__":
    # TapeContext manages AD lifecycle
    with TapeContext():
        # Input variables
        a = Variable(2.0)
        b = Variable(-3.0)
        c = Variable(10.0)
        x = Variable(-1.0)
        y = Variable(3.0)

        # Computation
        d1 = a * x
        d2 = b * y
        d3 = d1 + d2

        out = d3 + c

        # Perform the backward pass
        out.backward()

        # Print results (data and gradients)
        print("--- Input Variables ---")
        print(f"{a=}")
        print(f"{b=}")
        print(f"{c=}")
        print(f"{x=}")
        print(f"{y=}")
        print("--- Intermediates ---")
        print(f"{d1=}")
        print(f"{d2=}")
        print(f"{d3=}")
        print("--- Final Output")
        print(f"{out=}")

    # Tape is automatically cleaned up when exiting the 'with' block
```
### Output
```console
--- Input Variables ---
a=Variable(data=2.000e+00, grad=-1.000e+00)
b=Variable(data=-3.000e+00, grad=3.000e+00)
c=Variable(data=1.000e+01, grad=1.000e+00)
x=Variable(data=-1.000e+00, grad=2.000e+00)
y=Variable(data=3.000e+00, grad=-3.000e+00)
--- Intermediates ---
d1=Variable(data=-2.000e+00, grad=1.000e+00)
d2=Variable(data=-9.000e+00, grad=1.000e+00)
d3=Variable(data=-1.100e+01, grad=1.000e+00)
--- Final Output
out=Variable(data=-1.000e+00, grad=1.000e+00)
```

## C API Overview
The C API in ad_engine.h provides the core functionalities:
* `void ad_init_tape(size_t initial_capacity)`: Initializes/resets the global computation tape.
* `void ad_destroy_tape()`: Frees all memory associated with the current tape.
* `t_node* ad_create_node(float value)`: Creates a new leaf node (input variable) on the tape.
* `t_node* ad_add(t_node* a, t_node* b)`: Adds two Values.
* `t_node* ad_mul(t_node* a, t_node* b)`: Multiplies two Values.
* `void ad_backward(t_node* output_node, float seed_gradient)`: Performs the backward pass.
* `float ad_get_gradient(t_node* v)`: Retrieves the gradient of a t_node.
* `float ad_get_data(t_node* v)`: Retrieves the data (forward pass result) of a t_node.


(See ad_engine.h for complete signatures and ad_engine.c for implementations.)

## Limitations & Future Work
* Scalar Only: Currently only supports scalar operations. No direct tensor/matrix support.
* Limited Operations: The set of implemented mathematical operations is very basic.
* Error Handling: Minimal error handling in the C core.
* Performance: While in C, it's not yet optimized for extreme performance (e.g., SIMD, advanced memory management, CUDA).
* Global Tape: Uses a single global tape. For concurrent computations, context-specific tapes might be needed (though the Python TapeContext provides a good user-level abstraction for sequential use).

### Potential Future Enhancements:
* Add more mathematical operations (tanh, log, division, etc.).
* Basic tensor support (element-wise ops, dot product).
* More robust error checking and reporting.
* More sophisticated memory management for the tape.
* Benchmarking and performance optimizations.
* Python packaging (setup.py).
* Further available operations:
    *   Power (`x ** const_exponent`)
    *   Sine (`sin`)
    *   Exponential (`exp`)
    *   ReLU (`relu`)
    *   (Extend with more: `tanh`, `log`, division, etc.)
* Topological sorting of tape

## Contributing
Not currently in need of contributions but they are always welcome.

## License
This project is licensed under the MIT License. See the LICENSE file for details (or add the license text directly here if you don't have a separate file).