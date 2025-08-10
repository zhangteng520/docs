# CUDA 关键字笔记

## 1. __device__
`__device__` 用于修饰只能在设备（GPU）上执行的函数或变量。被 `__device__` 修饰的函数只能被设备端代码调用，不能被主机（CPU）端直接调用。

示例：
```cpp
__device__ int add(int a, int b) {
	return a + b;
}
```

## 2. __global__
`__global__` 用于修饰 GPU 上的内核函数（kernel），这些函数可以被主机端调用，并在设备上并行执行。调用方式为 `<<<...>>>`。

示例：
```cpp
__global__ void kernelFunc() {
	// ...
}
```

## 3. __host__
`__host__` 用于修饰只能在主机（CPU）上执行的函数。通常与 `__device__` 联合使用，表示函数可在主机和设备上都可用。

示例：
```cpp
__host__ __device__ int max(int a, int b) {
	return a > b ? a : b;
}
```

## 4. __shared__
`__shared__` 用于声明线程块内所有线程共享的变量，存储在共享内存中，适合线程间数据交换。

示例：
```cpp
__shared__ float sharedData[256];
```

## 5. __constant__
`__constant__` 用于声明常量内存变量，适合所有线程只读的数据，存储在设备的常量内存中。

示例：
```cpp
__constant__ float constData[128];
```

---

这些关键字是 CUDA 编程的基础，合理使用可提升 GPU 程序的性能和可读性。
