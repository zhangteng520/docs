# C++ Lambda 表达式笔记

C++11 引入了 Lambda 表达式（匿名函数），用于简洁地定义可调用对象，常用于算法、回调等场景。

## 基本语法
```cpp
[捕获列表](参数列表) -> 返回类型 {
	// 函数体
}
```

示例：
```cpp
auto add = [](int a, int b) { return a + b; };
int sum = add(2, 3); // sum = 5
```

## 捕获列表
- `[ ]`   不捕获任何变量
- `[=]`   以值捕获外部变量
- `[&]`   以引用捕获外部变量
- `[x, &y]`  x值捕获，y引用捕获

## 应用示例
```cpp
std::vector<int> v{1,2,3,4};
std::for_each(v.begin(), v.end(), [](int n){ std::cout << n << ", "; });
```

## 可变Lambda
使用 `mutable` 允许修改值捕获的变量副本：
```cpp
int x = 1;
auto f = [x]() mutable { x = 5; };
f(); // x外部不变
```

## Lambda类型
Lambda是匿名类的对象，可用 `auto` 声明，也可用 `std::function` 存储。

---

Lambda 表达式让 C++ 代码更简洁灵活，适合函数式编程风格。
