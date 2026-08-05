# Ascend NPU 算子代码审查报告

**文件**: `mat_mul_asw_kernel.h`  
**审查范围**: kernel实现、全局tensor类型、内存访问

---

## Bug 列表

### Bug #1: cGlobal_ 声明使用了错误的数据类型

- **位置**: 第49行
- **类型**: 全局Tensor类型错误
- **严重程度**: 严重（Critical）
- **描述**:  
  输出矩阵C的GlobalTensor声明为 `GlobalTensor<A_T> cGlobal_`，使用了输入矩阵A的元素类型 `A_T`，而非输出矩阵C应有的类型 `C_T`（即 `typename C_TYPE::T`）。在MatMul场景中，输入类型（如 `half`/`bf16`）与输出类型（如 `float`）通常不同。此错误导致：
  1. 第75行 `SetGlobalBuffer` 传入 `reinterpret_cast<__gm__ C_T *>(cGM)` 与容器声明类型 `A_T` 不匹配，产生隐式类型不安全转换或编译期类型推导错误。
  2. 第137行 `mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], ...)` 写出时使用错误的元素宽度计算偏移，导致内存越界访问或数据截断/写坏。
  
  **正确代码应为**:
  ```cpp
  GlobalTensor<C_T> cGlobal_;
  ```

- **触发条件**:  
  当 `A_TYPE::T`（输入矩阵A的数据类型）与 `C_TYPE::T`（输出矩阵C的数据类型）不同时必然触发。典型场景：A为 `half`（2字节），C为 `float`（4字节）。此时 `cGlobal_[offset]` 按 `half` 步长寻址，实际应按 `float` 步长寻址，导致偏移量错误减半，写入位置错乱，后续数据被覆盖。

- **测试方案**:  
  1. 构造测试用例：A_TYPE=MatmulType<half, ...>，C_TYPE=MatmulType<float, ...>，执行矩阵乘法。
  2. 对比输出矩阵C与CPU参考结果，验证数值正确性。
  3. 检查输出buffer是否有越界写入（使用内存检测工具如CANN的overflow detection）。
  4. 使用相同类型（A_T == C_T，如都是float）做对照组，此时bug不会显现。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第49行 | 全局Tensor类型错误 | Critical | `cGlobal_` 声明为 `GlobalTensor<A_T>` 应为 `GlobalTensor<C_T>`，导致输出矩阵偏移计算和内存访问类型错误 |
