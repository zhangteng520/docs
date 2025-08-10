# CUDA Thrust 库笔记

Thrust 是 NVIDIA 提供的 C++ 并行算法库，类似于 STL，简化了 CUDA 上的并行编程。

## 主要特性
- 提供常用算法：如 sort、reduce、transform、scan 等
- 支持主机（host）和设备（device）两种后端
- 容器类型：thrust::host_vector、thrust::device_vector
- 与原生 CUDA 互操作性好

## 基本用法
```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>

int main() {
	thrust::device_vector<int> d_vec(4);
	d_vec[0] = 3; d_vec[1] = 1; d_vec[2] = 4; d_vec[3] = 2;
	thrust::sort(d_vec.begin(), d_vec.end());
	// d_vec: 1, 2, 3, 4
	return 0;
}
```

## 常用算法示例
- 排序：`thrust::sort`
- 归约：`thrust::reduce`
- 变换：`thrust::transform`
- 扫描：`thrust::inclusive_scan`

## 容器类型
- `thrust::host_vector<T>`：主机端容器
- `thrust::device_vector<T>`：设备端容器

## 典型应用场景
- 大规模数据并行处理
- 快速原型开发
- 替代手写 CUDA kernel 的常见操作

---

Thrust 让 CUDA 并行编程更高效、易用，适合数据并行算法的快速实现。
