---
date: 2025-07-19
categories:
  - Linux
  - 性能优化
---

# vector

<!-- more -->


## vector的扩容机制

`std::vector`的扩容机制是其性能优化的关键。当元素数量超过当前容量时，vector会自动分配更大的内存空间，并将原有元素移动到新空间。

- **resize(n)**：调整vector的元素个数为n，若n大于当前size，则插入默认值元素；若小于则删除多余元素。
- **reserve(n)**：预留容量为n，不改变size，只影响capacity，避免多次扩容带来的性能损耗。

扩容通常会按2倍或1.5倍策略增长（具体实现依赖于STL库），每次扩容都需要将原有元素移动到新内存，涉及大量拷贝或移动操作。

```cpp
std::vector<int> v;
v.reserve(1000); // 预分配，减少扩容次数
for (int i = 0; i < 1000; ++i) v.push_back(i);
```

## vector对象的大小

`std::vector`对象本身只包含三个指针（或指针+size_t）：指向数据区的指针、size、capacity。其本身占用的内存很小，实际数据存储在堆上。

```cpp
std::vector<int> v(10);
std::cout << sizeof(v) << std::endl; // 通常为24字节（x64）
```

## 内存序

内存序（Endianness）指多字节数据在内存中的存储顺序。常见有大端（Big Endian）和小端（Little Endian）。

- x86/x64平台普遍采用小端序。
- 网络字节序为大端。

```cpp
union U { int i; char c[4]; } u = {0x12345678};
if (u.c[0] == 0x78) std::cout << "小端";
else std::cout << "大端";
```

## 右值引用和移动构造函数

右值引用（T&&）和移动构造函数极大提升了vector等容器的性能，避免了不必要的深拷贝。

```cpp
std::vector<std::string> v;
v.push_back("hello"); // 移动构造，避免拷贝
std::vector<std::string> v2 = std::move(v); // v变为空，资源转移到v2
```

vector扩容时，优先调用元素的移动构造函数（如果有），否则退化为拷贝构造。

## 进程的内存分区及ASLR 机制

现代操作系统下，进程的虚拟地址空间通常分为：

- 代码段（.text）
- 数据段（.data/.bss）
- 堆（heap）
- 栈（stack）
- 共享库映射区

ASLR（Address Space Layout Randomization，地址空间布局随机化）是一种安全机制，每次程序启动时，堆、栈、库等的基址都会随机变化，防止攻击者利用固定地址进行攻击。

## vector对象的大小

vector对象本身很小，通常只包含指向数据的指针、size和capacity等成员（如x64下24字节），实际数据存储在堆上。vector的大小与元素数量无关。

## 内存序

内存序（大小端）影响多字节数据的存储顺序。常见判断方法：

```cpp
int n = 0x12345678;
if (*(char*)&n == 0x78) std::cout << "小端序";
else std::cout << "大端序";
```

## 右值引用和移动构造函数

右值引用（T&&）和移动构造函数可极大提升vector等容器的性能，避免不必要的深拷贝。vector扩容或元素转移时，优先调用移动构造函数。

常见用法：
```cpp
std::vector<std::string> v;
v.push_back("abc"); // 移动构造
std::vector<std::string> v2 = std::move(v); // v资源转移到v2
```



