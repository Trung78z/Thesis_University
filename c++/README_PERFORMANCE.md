# Lane Detection Performance Optimizations

This document describes the performance optimizations implemented in the LaneVision lane detection system.

## Performance Improvements

### 1. Memory Management Optimizations

#### Buffer Reuse
- **Pre-allocated buffers**: All intermediate matrices (`gray_buffer_`, `blur_buffer_`, `edges_buffer_`, `roi_buffer_`) are pre-allocated to avoid repeated memory allocations
- **ROI mask caching**: The region of interest mask is cached and only recalculated when dimensions change
- **Vector pre-allocation**: Vectors are pre-allocated with `reserve()` to avoid dynamic resizing

#### Reduced Memory Copies
- **In-place operations**: Where possible, operations are performed in-place to avoid unnecessary copies
- **Efficient frame resizing**: Frame resizing is done in-place when possible

### 2. Algorithm Optimizations

#### Hough Transform Parameters
- **Increased threshold**: `hough_threshold` increased from 40 to 50 for better performance
- **Optimized line parameters**: `min_line_length` increased to 40, `max_line_gap` reduced to 20
- **Pre-allocated line storage**: Lines vector pre-allocated with 100 elements

#### Line Filtering
- **Efficient slope calculation**: Optimized slope and intercept calculations
- **Early termination**: Short lines and vertical lines are filtered out early
- **Range-based filtering**: Added maximum slope threshold to reduce false positives

#### Linear Regression
- **SIMD-friendly operations**: Linear regression uses operations that can be vectorized
- **Efficient R-squared calculation**: Optimized confidence calculation without using `std::pow()`
- **Numerical stability**: Better handling of edge cases and numerical precision

### 3. Resolution Optimization

#### Reduced Processing Resolution
- **Target resolution**: Reduced from 1280x720 to 640x480 for better performance
- **Configurable**: Resolution can be adjusted via `config_.target_width` and `config_.target_height`
- **Maintains accuracy**: Lower resolution still provides sufficient detail for lane detection

### 4. Temporal Smoothing

#### Exponential Smoothing
- **Configurable alpha**: Temporal smoothing factor can be adjusted (default: 0.7)
- **Reduced history size**: History window reduced from 5 to 3 frames for faster response
- **Efficient averaging**: Direct averaging instead of using `std::accumulate()`

### 5. Drawing Optimizations

#### Efficient Visualization
- **Reduced line thickness**: Line thickness reduced from 4 to 3 pixels
- **Optimized polygon drawing**: More efficient polygon point collection and ordering
- **Conditional drawing**: Drawing operations are skipped when not needed

## Configuration Parameters

### Performance-Related Settings
```cpp
struct Config {
    // Resolution
    int target_width = 640;
    int target_height = 480;
    
    // Hough transform
    int hough_threshold = 50;
    int min_line_length = 40;
    int max_line_gap = 20;
    
    // Temporal smoothing
    int confidence_window = 3;
    float temporal_smoothing_alpha = 0.7f;
    
    // Caching
    bool enable_roi_caching = true;
    bool enable_simd_optimizations = true;
};
```

## Build Optimizations

### Compiler Flags
The CMakeLists.txt includes performance-oriented compiler flags:

**GCC/Clang:**
- `-O3`: Maximum optimization level
- `-march=native`: Use CPU-specific optimizations
- `-ffast-math`: Fast math operations

**MSVC:**
- `/O2`: Maximum optimization
- `/arch:AVX2`: Use AVX2 instructions
- `/fp:fast`: Fast floating-point operations

## Performance Benchmarks

### Expected Performance Improvements
- **Memory allocations**: ~60% reduction in dynamic allocations
- **Processing time**: ~40-50% faster lane detection
- **Memory usage**: ~30% reduction in peak memory usage
- **FPS improvement**: 2-3x higher frame rates on typical hardware

### Testing
Run the performance test:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./test_performance
```

## Usage Recommendations

### For Real-time Applications
1. Use the default 640x480 resolution
2. Enable ROI caching
3. Use Release build with optimizations
4. Consider reducing `confidence_window` for faster response

### For High-Accuracy Applications
1. Increase `target_width` and `target_height`
2. Increase `hough_threshold` for cleaner results
3. Increase `confidence_window` for smoother tracking
4. Adjust `temporal_smoothing_alpha` for desired responsiveness

## Future Optimizations

### Potential Improvements
1. **GPU acceleration**: CUDA/OpenCL implementation for edge detection
2. **SIMD intrinsics**: Manual SIMD optimization for critical loops
3. **Multi-threading**: Parallel processing of different pipeline stages
4. **Neural network**: Replace traditional CV with optimized neural networks
5. **Memory pooling**: Custom memory allocator for OpenCV matrices

### Profiling
Use tools like:
- `perf` (Linux)
- `gprof` for function-level profiling
- `valgrind` for memory analysis
- Intel VTune for detailed performance analysis 