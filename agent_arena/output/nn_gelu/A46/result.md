# GeLU Kernel DAG CopyOut类型与注册一致性 代码审查报告

## Bug: CopyOut类型为T(float)而非U(输入/输出类型)，与算子注册不一致

- **位置**: `gelu_dag.h` 第78行
  ```cpp
  using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResultCast>;
  ```

- **描述**: 在 `GeluDAG` 模板中，`OpCopyOut` 使用的类型是 `T`（默认为float），而非 `U`（实际的输入输出dtype）。DAG流程为：
  - CopyIn: `Vec::CopyIn<U>` - 从GM搬入U类型数据
  - Cast: `Vec::Cast<T, U>` - 将U转为float
  - 计算: `GeluCustom<T>` - 在float精度下计算
  - Cast: `Vec::Cast<U, T, CAST_MODE_RINT>` - 将float转回U类型
  - CopyOut: `Vec::CopyOut<T>, Placeholder::Out0<T>` - **以T(float)类型写出到GM**

  问题在于：`OpResultCast` 的输出类型是 `U`（已经从float Cast回了U类型），但 `CopyOut` 声明为 `Vec::CopyOut<T>` 且输出placeholder为 `Placeholder::Out0<T>`。这导致：
  1. 输出GM buffer的类型声明为float，但算子注册中输出dtype为U（如float16/bfloat16），类型不匹配。
  2. 当 `U != T` 时（如U=half），OpResultCast产出的是half类型数据，但CopyOut期望float类型，造成类型不匹配的编译错误或运行时数据解释错误。
  3. 输出tensor的实际内存layout与算子注册声明的dtype不一致，下游算子读取时会按错误的dtype解释数据。

- **触发输入**: 输入dtype为float16或bfloat16，例如shape为 `[2, 512]`，dtype=float16。此时 `U=half, T=float`，CopyOut类型为float但实际输出应为float16。

- **预期异常**: 编译期可能因DAG类型推导矛盾而报模板实例化错误；若绕过编译，运行时输出GM中写入的数据格式与声明不符，导致输出数据完全错误（float数据被按float16解释），后续算子读取到乱码数据。
