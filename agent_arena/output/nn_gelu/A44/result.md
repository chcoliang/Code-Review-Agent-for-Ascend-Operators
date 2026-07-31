# GeLU Tiling (arch35) 代码审查报告

## Bug: UB Size 错误地乘以2

- **位置**: `gelu_tiling_arch35.cpp` 第138行
  ```cpp
  compileInfoPtr->ubSize = compileInfoPtr->ubSize * 2;
  ```

- **描述**: 在 `TilingPrepareForGelu` 函数中，通过 `ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize)` 获取UB大小后，将其乘以2。在Ascend 910B平台上，`GetCoreMemSize` API返回的已经是单个AIV核可使用的完整UB容量。将其乘以2会导致tiling计算时认为可用UB空间是实际的两倍，计算出过大的tile size，实际kernel执行时UB buffer溢出。

- **触发输入**: 任意合法的GeLU算子调用，例如输入shape为 `[8, 65536]`，dtype为float32。较大的tensor更容易触发，因为tiling基于错误的UB大小可能分配超出实际容量的buffer。

- **预期异常**: 运行时AICore UB内存越界访问，可能导致硬件异常（Bus Error）、计算结果错误（buffer相互覆盖），或设备报 `EZ9999` 错误。
