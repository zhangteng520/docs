---
date: 2024-01-31 
categories:
  - Share
---


# 代码坏的味道


代码坏味道（Code Smells）是指代码中可能导致维护困难、易出错或可读性差的结构和模式。坏味道并不一定是 bug，但通常预示着需要重构。
<!-- more -->

## 常见代码坏味道类型

### 1. 长函数（Long Function）
函数体过长，逻辑复杂，难以理解和维护。应拆分为更小的函数。

### 2. 过长类（Large Class）
类承担过多职责，代码臃肿。可通过提取类或分离职责优化。

### 3. 过多参数（Long Parameter List）
函数参数过多，调用困难。可用对象封装参数或减少依赖。

### 4. 重复代码（Duplicated Code）
同样的代码片段在多个地方出现。应提取为函数或类复用。

### 5. 过度嵌套（Deep Nesting）
if/for/while 等嵌套层级过深，影响可读性。可通过提前返回、拆分函数等方式优化。

### 6. 神秘命名（Mystery Name）
变量、函数、类命名不清晰，难以理解其用途。应使用有意义的名称。

### 7. 数据泥球（Data Clumps）
一组数据总是一起出现，建议封装为对象。

### 8. 发散式变化（Divergent Change）
一个类经常因不同原因被修改，说明职责不单一。

### 9. 霰弹式修改（Shotgun Surgery）
一个小改动需要在多个类或文件中做出修改，说明耦合度过高。

### 10. 依恋情结（Feature Envy）
一个类频繁访问另一个类的数据，说明功能划分不合理。

### 11. 过度注释（Excessive Comments）
代码需要大量注释才能理解，通常是代码本身不够清晰。

### 12. 基本类型偏执（Primitive Obsession）
过度使用基本类型而不是对象或枚举，导致代码难以扩展。

### 13. 过度耦合（Tight Coupling）
模块之间依赖过多，难以独立修改和测试。

### 14. 过度暴露（Data Class）
只包含字段和 getter/setter 的类，没有行为。

---


## 代码坏味道示例（C++）

```cpp
// 长函数示例
void processData(std::vector<int>& data) {
    // ...几十行处理逻辑...
    for (int i = 0; i < data.size(); ++i) {
        // ...
    }
    // ...
}

// 重复代码示例
int calcArea1(int w, int h) {
    return w * h;
}
int calcArea2(int w, int h) {
    return w * h;
}

// 神秘命名示例
int f(int x) {
    return x * 2; // x 是什么？
}

// 数据泥球示例
void printPerson(std::string name, int age, std::string address) {
    std::cout << name << age << address << std::endl;
}

// 过度暴露（Data Class）示例
class Person {
public:
    std::string name;
    int age;
    std::string address;
    // 只有数据，没有行为
};
```

---

## 如何处理代码坏味道
- 重构（Refactoring）：通过重命名、拆分函数/类、消除重复等方式优化代码结构。
- 编写单元测试，确保重构后功能不变。
- 代码评审，团队共同发现和消除坏味道。