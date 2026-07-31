# Softmax V2 Base Tiling 代码审查报告

## Bug: UB Size 错误地乘以2

- **位置**: `softmax_v2_base_tiling.cpp` 第281行
  ```cpp
  compileInfoPtr->ubSize = static_cast<int64_t>(ubSizeTemp) * 2;
  ```

- **描述**: 在 `TilingPrepareForSoftmaxV2AscendC` 函数中，通过 `ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp)` 获取UB大小后，将其乘以2存入 `compileInfoPtr->ubSize`。在Ascend 910B平台上，`GetCoreMemSize` 返回的已经是每个AIV核可用的完整UB大小（192KB）。将其乘以2会导致计算出的UB容量为实际的两倍（384KB），后续tiling计算会分配超出物理UB容量的buffer，导致数据越界写入或硬件异常。

- **触发输入**: 任意合法的SoftmaxV2算子调用，例如输入shape为 `[32, 1024]`，dtype为float16，axes=[-1]。当tiling逻辑基于错误的UB大小（2倍）计算切分参数时，实际kernel运行时会因UB空间不足而越界。

- **预期异常**: 运行时可能出现UB内存越界访问，触发AICore硬件异常（如Bus Error），或产生计算结果错误（数据被覆盖）。在严重情况下可能导致设备挂死或报 `EZ9999` 错误。
