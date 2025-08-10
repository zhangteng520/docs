# CUDA 线程ID计算笔记

在 CUDA 编程中，线程 ID 的计算是并行算法实现的基础。每个线程都有唯一的标识符，可用于数据分配和任务划分。

## 1. 基本概念
- `threadIdx.x/y/z`：线程在 block 内的索引
- `blockIdx.x/y/z`：block 在 grid 内的索引
- `blockDim.x/y/z`：每个 block 的线程数
- `gridDim.x/y/z`：grid 中 block 的数量

## 2. 一维线程ID计算
```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```
`tid` 即全局唯一线程编号，常用于一维数据处理。

## 3. 二维线程ID计算
```cpp
int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
```
适用于二维数据（如图像）。

## 4. 三维线程ID计算
```cpp
int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
int tid_z = blockIdx.z * blockDim.z + threadIdx.z;

int tid = tid_x 
        + tid_y * (gridDim.x * blockDim.x) 
        + tid_z * (gridDim.x * blockDim.x * gridDim.y * blockDim.y);
```
适用于三维数据处理。

## 5. 示例
```cpp
__global__ void kernel() {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 使用 tid 进行数据处理
}
```

---

合理计算线程ID是实现高效并行的关键。
