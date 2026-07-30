# 算子编译与 NPU 部署指南

## 环境信息

| 项目 | 值 |
|------|------|
| NPU | Ascend910_9382 (ascend910_93 平台) |
| CANN Runtime | 8.5.0 |
| CANN Toolkit | 8.5.0 (含 bisheng 编译器) |
| bisheng 路径 | `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng` |
| ops-math 源码 | 本仓库 `ops-math/` 目录 (8.5.0 分支) |
| 编译输出路径 | `/dev/shm/` (tmpfs，避免 NPU 设备文件权限问题) |

## NPU 在线编译执行环境（关键！）

**要让算子在 NPU AICore 上真正执行计算，必须设置以下环境变量：**

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$ASCEND_HOME_PATH/compiler/bin:$PATH
export PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export ASCEND_OP_COMPILER_CACHE_MODE=enable
export ASCEND_OP_COMPILER_CACHE_DIR=/dev/shm/op_cache
mkdir -p /dev/shm/op_cache
```

**验证成功：**
```
$ ./A10_bug1
aclnnMul return code: 0
Results: [2.0, 6.0, 12.0, 20.0]    ← NPU AICore 计算结果正确！
Expected: [2.0, 6.0, 12.0, 20.0]
```

## 前提条件

1. CANN 8.5.0 runtime 已安装
2. CANN 8.5.0 toolkit 已安装（含 bisheng 编译器）
3. NPU 驱动正常（`npu-smi info` 正常）
4. `mul_def.cpp` 中已添加 `ascend910_93` 配置：
```cpp
this->AICore().AddConfig("ascend910_93", aicoreConfig);
```

## 编译步骤

### 第一步：设置环境变量

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH
```

### 第二步：在 /dev/shm 中执行 cmake

**关键**：必须在 tmpfs (`/dev/shm`) 下执行 cmake，否则 bisheng 编译器产生的 `.__dpc*` 设备文件在普通文件系统上会因权限问题导致 cmake 失败。

```bash
rm -rf /dev/shm/ops_build && mkdir /dev/shm/ops_build && cd /dev/shm/ops_build

OPS_MATH=<项目路径>/ops-math

cmake $OPS_MATH \
  -DASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH \
  -DASCEND_COMPUTE_UNIT="ascend910_93" \
  -DCUSTOM_VENDOR_NAME=custom_test
```

预期输出：
```
-- Configuring done
-- Generating done
-- Build files have been written to: /dev/shm/ops_build
```

### 第三步：编译（仅 Mul 相关 target）

```bash
make opapi_math ophost_math mul_src_copy -j8
```

编译时间约 3-4 分钟，产出：
- `libopapi_math.so` — op_api 层共享库（含 aclnnMul 参数校验逻辑）
- `libophost_math.so` — op_host 层共享库（含 tiling 逻辑）
- `tbe/ascendc/mul/mul_apt.cpp` — kernel 源码（用于在线编译）
- `tbe/ascendc/mul/arch35/mul_dag.h` — kernel DAG 定义

### 第四步：部署到系统 vendors 目录

**需要 root 权限**：

```bash
VENDOR=/usr/local/Ascend/cann-8.5.0/opp/vendors/custom_test
BUILD=/dev/shm/ops_build

sudo mkdir -p $VENDOR/op_api/lib
sudo mkdir -p $VENDOR/op_impl/ai_core/tbe/ascendc
sudo cp $BUILD/libopapi_math.so $VENDOR/op_api/lib/libcust_opapi.so
sudo cp $BUILD/libophost_math.so $VENDOR/op_impl/ai_core/tbe/
sudo cp -r $BUILD/tbe/ascendc/mul $VENDOR/op_impl/ai_core/tbe/ascendc/
```

### 第五步：验证 NPU 执行

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/lib64:...
./npu_tests/A10_bug1
```

预期输出（部署成功后）：
```
aclnnMulGetWorkspaceSize: 0
aclnnMul: 0 (SUCCESS)
Results: [2.0, 6.0, 12.0, 20.0]
NPU COMPUTE SUCCESS!
```

## 编译修改后的（buggy）算子

要编译注入了 bug 的版本，替换 ops-math 中的源文件后重新执行步骤 2-4：

```bash
# 例: 注入 A07 的 bug (错误码伪装)
cp agent_arena/cases/op_api/A07/aclnn_mul.cpp ops-math/math/mul/op_api/aclnn_mul.cpp

# 重新编译
cd /dev/shm/ops_build
make opapi_math -j8

# 重新部署
sudo cp /dev/shm/ops_build/libopapi_math.so $VENDOR/op_api/lib/libcust_opapi.so

# 运行测试
./npu_tests/A07_bug1
# 预期: SEGFAULT (因为错误码被伪装为 SUCCESS)
```

## 恢复正确版算子

```bash
cd ops-math
git checkout -- math/mul/op_api/aclnn_mul.cpp
cd /dev/shm/ops_build && make opapi_math -j8
sudo cp /dev/shm/ops_build/libopapi_math.so $VENDOR/op_api/lib/libcust_opapi.so
```

## 常见问题

### Q: cmake 报 "could not be removed: .__dpc*"
**A**: 必须在 `/dev/shm` (tmpfs) 下执行 cmake。普通文件系统上 bisheng 产生的设备控制文件需要特殊权限删除。

### Q: aclnnMul 返回 561000/561103
**A**: kernel binary 没有正确部署。需要将算子安装到系统 vendors 目录（需 root），或使用在线编译机制。

### Q: 编译时 "GenerateEsPackage not found"
**A**: ops-math 8.5.0 需要 CANN toolkit 8.5.0。8.3 的 toolkit 不兼容。

### Q: "No known features for CXX compiler" 
**A**: 不要使用 `-DCMAKE_CXX_COMPILER_FORCED=TRUE`，这会跳过 g++ 特性检测。
