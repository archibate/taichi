# Changelog
- (Feb  12, 2020) v0.4.6 released.
   - (For compiler developers) An error will be raised when `TAICHI_REPO_DIR` is not a valid path (by **Yubin Peng [archibate]**)
   - Fixed a CUDA backend deadlock bug
   - Added test selectors `ti.require()` and `ti.archs_excluding()` (by **Ye Kuang [k-ye]**)
   - `ti.init(**kwargs)` now takes a parameter `debug=True/False`, which turns on debug mode if true
   - ... or use `TI_DEBUG=1` to turn on debug mode non-intrusively
   - Fixed `ti.profiler_clear`
   - Added `GUI.line(begin, end, color, radius)` and `ti.rgb_to_hex`
   - 
   - Renamed `ti.trace` (Matrix trace) to `ti.tr`. `ti.trace` is now for logging with `ti.TRACE` level 
   - Fixed return value of `ti test_cpp` (thanks to **Ye Kuang [k-ye]**)
   - Raise default loggineg level to `ti.INFO` instead of trace to make the world quiter
   - General performance/compatibility improvements
   - Doc updated
- (Feb   6, 2020) v0.4.5 released.
   - **`ti.init(arch=..., print_ir=..., default_fp=..., default_ip=...)`** now supported. `ti.cfg.xxx` is deprecated
   - **Immediate data layout specification** supported after `ti.init`. No need to wrap data layout definition with `@ti.layout` anymore (unless you intend to do so)
   - `ti.is_active`, `ti.deactivate`, `SNode.deactivate_all` supported in the new LLVM x64/CUDA backend. [Example](https://github.com/taichi-dev/taichi/blob/8b575a8ec2d8c7112191eef2a8316b793ba2452d/examples/taichi_sparse.py) <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/taichi/sparse_grids.gif">
   - Experimental [Windows non-UTF-8 path](https://github.com/taichi-dev/taichi/issues/428) fix (by **Yubin Peng [archibate]**)
   - `ti.global_var` (which duplicates `ti.var`) is removed
   - `ti.Matrix.rotation2d(angle)` added
- (Feb   5, 2020) v0.4.4 released.
   - For developers: [ffi-navigator](https://github.com/tqchen/ffi-navigator) support [[doc](https://taichi.readthedocs.io/en/latest/contributor_guide.html#efficient-code-navigation-across-python-c)]. (by **masahi**)
   - Fixed `f64` precision support of `sin` and `cos` on CUDA backends (by **Kenneth Lozes [KLozes]**)
   - Make Profiler print the arch name in its title (by **Ye Kuang [k-ye]**)
   - Tons of invisible contributions by **Ye Kuang [k-ye]**, for the WIP Metal backend
   - `Profiler` working on CPU devices. To enable, `ti.cfg.enable_profiler = True`. Call `ti.profiler_print()` to print kernel running times
   - General performance improvements
- (Feb   3, 2020) v0.4.3 released.
   - `GUI.circles` 2.4x faster
   - General performance improvements
- (Feb   2, 2020) v0.4.2 released.
   - GUI framerates are now more stable
   - Optimized OffloadedRangeFor with const bounds. Light computation programs such as `mpm88.py` is 30% faster on CUDA due to reduced kernel launches
   - Optimized CPU parallel range for performance
- (Jan  31, 2020) v0.4.1 released.
   - **Fixed an autodiff bug introduced in v0.3.24. Please update if you are using Taichi differentiable programming.**
   - Updated `Dockerfile` (by **Shenghang Tsai [jackalcooper]**)
   - `pbf2d.py` visualization performance boosted (by **Ye Kuang [k-ye]**)
   - Fixed `GlobalTemporaryStmt` codegen
- (Jan  30, 2020) v0.4.0 released.
   - Memory allocator redesigned
   - Struct-fors with pure dense data structures will be demoted into a range-for, which is faster since no element list generation is needed
   - Python 3.5 support is dropped. Please use Python 3.6(pip)/3.7(pip)/3.8(Windows: pip; OS X & Linux: build from source) (by **Chujie Zeng [Psycho7]**)
   - `ti.deactivate` now supported on sparse data structures
   - `GUI.circles` (batched circle drawing) performance improved by 30x
   - Minor bug fixes (by **Yubin Peng [archibate], Ye Kuang [k-ye]**)
   - Doc updated
- (Jan  20, 2020) v0.3.25 released.
   - Experimental [CPU-only support for NVIDIA Jetson Nano](https://user-images.githubusercontent.com/34827518/72769070-62b1b200-3c34-11ea-8f6e-0f339b5b09ca.jpg) (with ARM CPUs. Building from source required.) (thanks to **Walter liu
 [hgnan]**)
- (Jan  19, 2020) v0.3.24 released.
   - `%` and `//` now follow Python semantics. Use `ti.raw_mod` for C-style `%` semantics (by **Chujie Zeng [Psycho7]**)
   - Parallel range-fors now supports non-compile-time constant bounds. For example, `for i in range(bound[0])` is supported
- (Jan  18, 2020) v0.3.23 released.
   - Taichi kernel calls now releases Python GIL
- (Jan  17, 2020) v0.3.22 released.
   - `ti.atomic_add()` now returns the old value (by **Ye Kuang [k-ye]**)
   - Experimental patch to Windows systems with malformed BIOS info (by **Chujie Zeng [Psycho7]**)
   - `ti.__version__` now returns the version triple, e.g. `(0, 3, 22)`
   - Fixed a CPU multithreading bug
   - Avoid accessor IR printing when setting `ti.cfg.print_ir = True`
   - Added `ti.cfg.print_accessor_ir`
   - Removed dependency on x86_64 SIMD intrinsics
   - Improved doc
- (Jan  11, 2020) v0.3.21 released.
   - GUI fix for OS X 10.14 and 10.15 (by **Ye Kuang [k-ye]**).
   - Minor improvements on documentation and profiler
- (Jan   2, 2020) v0.3.20 released.
   - Support `ti.static(ti.grouped(ti.ndrange(...)))`
- (Jan   2, 2020) v0.3.19 released.
   - Added `ti.atan2(y, x)`
   - Improved error msg when using float point numbers as tensor indices
- (Jan   1, 2020) v0.3.18 released.
   - Added `ti.GUI` class
   - Improved the performance of performance `ti.Matrix.fill`
- (Dec  31, 2019) v0.3.17 released.
   - Fixed cuda context conflict with PyTorch  (thanks to @Xingzhe He for reporting)
   - Support `ti.Matrix.T()` for transposing a matrix
   - Iteratable `ti.static(ti.ndrange)`
   - Fixed `ti.Matrix.identity()`
   - Added `ti.Matrix.one()` (create a matrix with 1 as all the entries)
   - Improved `ir_printer` on SNodes
   - Better support for `dynamic` SNodes.
     - `Struct-for's` on `dynamic` nodes supported
     - `ti.length` and `ti.append` to query and manipulate dynamic nodes
- (Dec  29, 2019) v0.3.16 released.
   - Fixed ndrange-fors with local variables (thanks to Xingzhe He for reporting this issue)
- (Dec  28, 2019) v0.3.15 released.
   - Multi-dimensional parallel range-for using `ti.ndrange`:
```python
  @ti.kernel
  def fill_3d():
    # Parallelized for all 3 <= i < 8, 1 <= j < 6, 0 <= k < 9
    for i, j, k in ti.ndrange((3, 8), (1, 6), 9):
      x[i, j, k] = i + j + k
```
- (Dec  28, 2019) v0.3.14 released.
   - GPU random number generator support for more than 1024x1024 threads
   - Parallelized element list generation on GPUs. Struct-fors significantly sped up.
   - `ti` and `tid` (debug mode) CLI commands
- (Dec  26, 2019) v0.3.13 released.
   - `ti.append` now returns the list length before appending
   - Fixed for loops with 0 iterations
   - Set `ti.get_runtime().set_verbose_kernel_launch(True)` to log kernel launches
   - Distinguish `/` and `//` following the Python convention
   - Allow using local variables as kernel argument type annotations
- (Dec  25, 2019) v0.3.11 released.
   - Support multiple kernels with the same name, especially in the OOP cases where multiple member kernels share the same name
   - Basic `dynamic` node support (`ti.append`, `ti.length`) in the new LLVM backend
   - Fixed struct-for loops on 0-D tensors
- (Dec  24, 2019) v0.3.10 released.
   - `assert <condition>` statement supported in Taichi kernels.
   - Comparison operator chaining (e.g. `1 < x <3`) supported in Taichi kernels.
- (Dec  24, 2019) v0.3.9 released.
   - `ti.classfunc` decorator for functions within a `data_oriented` class
   - `[Expr/Vector/Matrix].to_torch` now has a extra argument `device`, which specifies the device placement for returned torch tensor, and should have type `torch.device`. Default=`None`.
   - Cross-device (CPU/GPU) taichi/PyTorch interaction support, when using `to_torch/from_torch`.
   - #kernels compiled during external array IO significantly reduced (from `matrix size` to `1`)
- (Dec  23, 2019) v0.3.8 released.
   - **Breaking change**: `ti.data_oriented` decorator introduced. Please decorate all your Taichi data-oriented objects using this decorator. To invoke the gradient versions of `classmethod`, for example, `A.forward`, simply use `A.forward.grad()` instead of `A.forward(__gradient=True)` (obsolete).
- (Dec  22, 2019) v0.3.5 released.
   - Maximum tensor dimensionality is 8 now (used to be 4). I.e., you can now allocate up to 8-D tensors.
- (Dec  22, 2019) v0.3.4 released.
   - 2D and 3D polar decomposition (`R, S = ti.polar_decompose(A, ti.f32)`) and svd (`U, sigma, V = ti.svd(A, ti.f32)`) support. Note that `sigma` is a `3x3` diagonal matrix.
   - Fixed documentation versioning
   - Allow `expr_init` with `ti.core.DataType` as inputs, so that `ti.core.DataType` can be used as `ti.func` parameter
- (Dec  20, 2019) v0.3.3 released.
   - Loud failure message when calling nested kernels. Closed #310
   - `DiffTaichi` examples moved to [a standalone repo](https://github.com/yuanming-hu/difftaichi)
   - Fixed documentation versioning
   - Correctly differentiating kernels with multiple offloaded statements
- (Dec  18, 2019) v0.3.2 released
   - `Vector.norm` now comes with a parameter `eps` (`=0` by default), and returns `sqrt(\sum_i(x_i ^ 2) + eps)`. A non-zero `eps` safe guards the operator's gradient on zero vectors during differentiable programming.
- (Dec  17, 2019) v0.3.1 released.
   - Removed dependency on `glibc 2.27`
- (Dec  17, 2019) v0.3.0 released.
   - Documentation significantly improved
   - `break` statements supported in while loops
   - CPU multithreading enabled by default
- (Dec  16, 2019) v0.2.6 released.
   - `ti.GUI.set_image(np.ndarray/Taichi tensor)`
   - Inplace adds are *atomic* by default. E.g., `x[i] += j` is equivalent to `ti.atomic_add(x[i], j)`
   - `ti.func` arguments are forced to pass by value
   - `min/max` can now take more than two arguments, e.g. `max(a, b, c, d)`
   - Matrix operators `transposed`, `trace`, `polar_decompose`, `determinant` promoted to `ti` scope. I.e., users can now use `ti.transposed(M)` instead of `ti.Matrix.transposed(M)`
   - `ti.get_runtime().set_verbose(False)` to eliminate verbose outputs
   - LLVM backend now supports multithreading on CPUs
   - LLVM backend now supports random number generators (`ti.random(ti.i32/i64/f32/f64`)
- (Dec  5, 2019) v0.2.3 released.
   - Simplified interaction between `Taichi`, `numpy` and `PyTorch`
     - `taichi_scalar_tensor.to_numpy()/from_numpy(numpy_array)`
     - `taichi_scalar_tensor.to_torch()/from_torch(torch_array)`
- (Dec  4, 2019) v0.2.2 released.
   - Argument type `ti.ext_arr()` now takes PyTorch tensors
- (Dec  3, 2019) v0.2.1 released.
   - Improved type mismatch error message
   - native `min`/`max` supprt
   - Tensor access index dimensionality checking
   - `Matrix.to_numpy`, `Matrix.zero`, `Matrix.identity`, `Matrix.fill`
   - Warning instead of error on lossy stores
   - Added some initial support for cross-referencing local variables in different offloaded blocks.
- (Nov 28, 2019) v0.2.0 released.
   - More friendly syntax error when passing non-compile-time-constant values to `ti.static`
   - Systematically resolved the variable name resolution [issue](https://github.com/yuanming-hu/taichi/issues/282)
   - Better interaction with numpy:
     - `numpy` arrays passed as a `ti.ext_arr()` [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_numpy.py)
       - `i32/f32/i64/f64` data type support for numpy
       - Multidimensional numpy arrays now supported in Taichi kernels
     - `Tensor.to_numpy()` and `Tensor.from_numpy(numpy.ndarray)` supported [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_cvt_numpy.py)
     - Corresponding PyTorch tensor interaction will be supported very soon. Now only 1D f32 PyTorch tensors supproted when using `ti.ext_arr()`. Please use numpy arrays as intermediate buffers for now
   - Indexing arrays with an incorrect number of indices now results in a syntax error
   - Tensor shape reflection: [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_tensor_reflection.py)
     - `Tensor.dim()` to retrieve the dimensionality of a global tensor
     - `Tensor.shape()` to retrieve the shape of a global tensor
     - Note the above queries will cause data structures to be materialized
   - `struct-for` (e.g. `for i, j in x`) now supports iterating over tensors with non power-of-two dimensions
   - Handy tensor filling: [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_fill.py)
     - `Tensor.fill(x)` to set all entries to `x`
     - `Matrix.fill(x)` to set all entries to `x`, where `x` can be a scalar or `ti.Matrix` of the same size
   - Reduced python package size
   - `struct-for` with grouped indices for better metaprogramming, especially in writing dimensionality-independent code, in e.g. physical simulation: [[examples]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_grouped.py)
```python
for I in ti.grouped(x): # I is a vector of size x.dim() and data type i32
  x[I] = 0

# If tensor x is 2D
for I in ti.grouped(x): # I is a vector of size x.dim() and data type i32
  y[I + ti.Vector([0, 1])] = I[0] + I[1]
# is equivalent to
for i, j in x:
  y[i, j + 1] = i + j
```

- (Nov 27, 2019) v0.1.5 released.
   - [Better modular programming support](https://github.com/yuanming-hu/taichi/issues/282)
   - Disalow the use of `ti.static` outside Taichi kernels
   - Documentation improvements (WIP)
   - Codegen bug fixes
   - Special thanks to Andrew Spielberg and KLozes for bug report and feedback.
- (Nov 22, 2019) v0.1.3 released.
   - Object-oriented programming. [[Example]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_oop.py)
   - native Python function translation in Taichi kernels:
     - Use `print` instead of `ti.print`
     - Use `int()` instead of `ti.cast(x, ti.i32)` (or `ti.cast(x, ti.i64)` if your default integer precision is 64 bit)
     - Use `float()` instead of `ti.cast(x, ti.f32)` (or `ti.cast(x, ti.f64)` if your default float-point precision is 64 bit)
     - Use `abs` instead of `ti.abs`
     - Use `ti.static_print` for compile-time printing

- (Nov 16, 2019) v0.1.0 released. Fixed PyTorch interface.
- (Nov 12, 2019) v0.0.87 released.
   - Added experimental Windows support with a [[known issue]](https://github.com/yuanming-hu/taichi/issues/251) regarding virtual memory allocation, which will potentially limit the scalability of Taichi programs (If you are a Windows expert, please let me know how to solve this. Thanks!). Most examples work on Windows now.
   - CUDA march autodetection;
   - [Complex kernel](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_complex_kernels.py) to override autodiff.
 - (Nov 4, 2019) v0.0.85 released.
   - `ti.stop_grad` for stopping gradients during backpropagation. [[Example]](https://github.com/yuanming-hu/taichi/blob/master/tests/python/test_stop_grad.py#L75);
   - Compatibility improvements on Linux and OS X;
   - Minor bug fixes.

<!---
| **Linux, Mac OS X** | **Windows** | Doc (WIP) | **Chat** |
|---------------------|------------------|----------------|------------------|
|[![Build Status](https://travis-ci.org/yuanming-hu/taichi.svg?branch=master)](https://travis-ci.org/yuanming-hu/taichi)|[![Build Status](https://ci.appveyor.com/api/projects/status/github/yuanming-hu/taichi?branch=master&svg=true)](https://ci.appveyor.com/project/IteratorAdvance/taichi)|[![Documentation Status](https://readthedocs.org/projects/taichi/badge/?version=latest)](http://taichi.readthedocs.io/en/latest/?badge=latest)|[![Join the chat at https://gitter.im/taichi-dev/Lobby](https://badges.gitter.im/taichi-dev/Lobby.svg)](https://gitter.im/taichi-dev/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)|
--->


## The Legacy Taichi Library [[Legacy branch]](https://github.com/yuanming-hu/taichi/tree/legacy)
The legacy **Taichi** library is an open-source computer graphics library written in C++14 and wrapped friendly with Python. It is no longer maintained since we have switched to the Taichi programming language and compiler.

## News
 - May 17, 2019: [Giga-Voxel SPGrid Topology Optimization Solver](https://github.com/yuanming-hu/spgrid_topo_opt) is released!
 - March 4, 2019: [MLS-MPM/CPIC solver](https://github.com/yuanming-hu/taichi_mpm) is now MIT-licensed!
 - August 14, 2018: [MLS-MPM/CPIC solver](https://github.com/yuanming-hu/taichi_mpm) reloaded! It delivers 4-14x performance boost over the previous state of the art on CPUs.

<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/topopt/bird-beak.gif">

 <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/water_wheel.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/sand_paddles.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/armodillo.gif" style=""> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/debris_flow.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/sand-sweep.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/sand_stir.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/bunny.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/robot_forward.gif">
<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/banana.gif"> <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/mls-mpm-cpic/cheese.gif">

### [Getting Started (Legacy)](https://taichi.readthedocs.io/en/latest/installation.html#)
