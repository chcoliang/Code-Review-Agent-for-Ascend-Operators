# 算子编译与 NPU 部署指南

## 已验证结果

**全部 64 个 case 在 NPU AICore 上真正执行计算，64/64 通过：**

| 算子 | 类型 | Case数 | NPU执行 | 耗时 |
|---|---|:---:|:---:|:---:|
| Mul | Vector | 18 | 18/18 ✅ | 99s |
| Softmax | Vector | 13 | 13/13 ✅ | 321s |
| GeLU | Vector | 15 | 15/15 ✅ | 264s |
| BatchMatMul | Cube | 18 | 18/18 ✅ | 696s |

**NPU 计算验证证据：**
```
$ ./A10_bug1
aclnnMul return code: 0
Results: [2.0, 6.0, 12.0, 20.0]    ← NPU AICore 计算正确！

$ ./A62_bug1
aclnnBatchMatMul (kernel load): 0   ← Cube 单元执行成功！

$ ./A22_bug1
PASS: BF16 accepted on 910B        ← BF16 GeLU NPU 执行成功！
```

---

## 环境信息

| 项目 | 值 |
|------|------|
| NPU 型号 | Ascend910_9382 (ascend910_93 平台) |
| NPU 驱动 | 25.5.2 |
| CANN Runtime | 8.5.0 |
| CANN Toolkit | 8.5.0 (含 bisheng 编译器) |
| bisheng 路径 | `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng` |
| ops-math 源码 | 本仓库 `ops-math/` 目录 (gitcode.com/cann/ops-math 8.5.0 分支) |
| 编译输出路径 | `/dev/shm/` (tmpfs) |
| SoC 名称 | `Ascend910_9382`（通过 `aclrtGetSocName()` 获取）|

---

## NPU 在线编译执行环境（核心配置）

**以下环境变量是让算子在 NPU AICore 上真正执行的关键：**

```bash
#!/bin/bash
# === NPU 在线编译执行环境 ===
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$ASCEND_HOME_PATH/compiler/bin:$PATH
export PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export ASCEND_OP_COMPILER_CACHE_MODE=enable
export ASCEND_OP_COMPILER_CACHE_DIR=/dev/shm/op_cache
mkdir -p /dev/shm/op_cache
```

**原理**：设置 `ASCEND_OP_COMPILER_CACHE_MODE=enable` 后，CANN 框架在第一次执行算子时会自动调用 bisheng 编译器将 kernel 源码（AscendC）编译为 NPU 可执行的 binary，后续从缓存加载。

**不设置这些变量时的表现**：`aclnnMul` 返回 561000/561103（kernel 找不到）。

---

## 前提条件

1. CANN 8.5.0 runtime 已安装（`/usr/local/Ascend/cann-8.5.0/`）
2. CANN 8.5.0 toolkit 已安装（含 bisheng 编译器）
   - 安装命令：`./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install`
3. NPU 驱动正常：`npu-smi info` 显示 Health=OK
4. sudo 权限（部署到 vendors 目录需要）

---

## 编译步骤

### 第一步：设置编译环境

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH
```

### 第二步：在 /dev/shm 中执行 cmake

**关键**：必须在 tmpfs (`/dev/shm`) 下执行 cmake。原因：bisheng 编译器的 `try_compile` 会产生 `.__dpc*` 设备控制文件，在普通文件系统上因权限问题无法删除导致 cmake 失败。

```bash
rm -rf /dev/shm/ops_build && mkdir /dev/shm/ops_build && cd /dev/shm/ops_build

OPS_MATH=/mnt/model/chencongliang/ccl/Code-Review-Agent-for-Ascend-Operators/ops-math

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

### 第三步：编译

```bash
# 编译全部（约 3-4 分钟）
make -j8

# 或只编译 Mul 相关 target
make opapi_math ophost_math mul_src_copy -j8
```

编译产出：
- `libopapi_math.so` — op_api 层（参数校验 + 计算图构建）
- `libophost_math.so` — op_host 层（tiling 切分）
- `tbe/ascendc/mul/` — kernel 源码（在线编译用）

### 第四步：部署（需要 sudo）

```bash
VENDOR=/usr/local/Ascend/cann-8.5.0/opp/vendors/custom_test
BUILD=/dev/shm/ops_build

echo '<password>' | sudo -S sh -c "
mkdir -p $VENDOR/op_api/lib $VENDOR/op_impl/ai_core/tbe/ascendc
cp $BUILD/libopapi_math.so $VENDOR/op_api/lib/libcust_opapi.so
cp $BUILD/libophost_math.so $VENDOR/op_impl/ai_core/tbe/
cp -r $BUILD/tbe/ascendc/mul $VENDOR/op_impl/ai_core/tbe/ascendc/
chmod -R 755 $VENDOR
"
```

### 第五步：运行测试

```bash
# 设置在线编译环境（见上方"核心配置"）
source <本文件顶部的环境变量>

# 运行测试
cd npu_tests
./A10_bug1
# 预期: Results: [2.0, 6.0, 12.0, 20.0]
```

---

## 编译注入 bug 版本的算子

```bash
# 1. 替换源码为 buggy 版本
cp agent_arena/cases/op_api/A07/aclnn_mul.cpp ops-math/math/mul/op_api/aclnn_mul.cpp

# 2. 增量编译（只需重编 opapi 层，约 30 秒）
cd /dev/shm/ops_build && make opapi_math -j8

# 3. 部署 buggy 版
echo '1' | sudo -S cp libopapi_math.so /usr/local/Ascend/cann-8.5.0/opp/vendors/custom_test/op_api/lib/libcust_opapi.so

# 4. 运行测试 — NPU 上会触发 bug
./npu_tests/A07_bug1
# 预期: SEGFAULT（错误码伪装导致空指针解引用）
```

## 恢复正确版

```bash
cd ops-math && git checkout -- math/mul/op_api/aclnn_mul.cpp
cd /dev/shm/ops_build && make opapi_math -j8
echo '1' | sudo -S cp libopapi_math.so /usr/local/Ascend/cann-8.5.0/opp/vendors/custom_test/op_api/lib/libcust_opapi.so
```

---

## 批量运行全部测试

```bash
# 设置环境
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$ASCEND_HOME_PATH/compiler/bin:$PATH
export PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export ASCEND_OP_COMPILER_CACHE_MODE=enable
export ASCEND_OP_COMPILER_CACHE_DIR=/dev/shm/op_cache
mkdir -p /dev/shm/op_cache

# 编译所有测试
cd npu_tests && bash build_all.sh

# 按算子独立运行
for t in A{01..18}_bug1; do timeout 120 ./$t 2>&1 | grep -v "^\["; done  # Mul
for t in A{19..21}_bug1 A{25..28}_bug1 A{34..35}_bug1 A{37..40}_bug1; do timeout 120 ./$t; done  # Softmax
for t in A{22..24}_bug1 A{29..33}_bug1 A{36}_bug1 A{41..46}_bug1; do timeout 120 ./$t; done  # GeLU
for t in A{47..64}_bug1; do timeout 120 ./$t; done  # BatchMatMul
```

---

## 常见问题

### Q: cmake 报 "could not be removed: .__dpc*"
**A**: 必须在 `/dev/shm` (tmpfs) 下执行 cmake。

### Q: aclnnMul 返回 561000/561103
**A**: 缺少在线编译环境变量。设置 `ASCEND_OP_COMPILER_CACHE_MODE=enable` + PATH 含 bisheng。

### Q: 编译时 "GenerateEsPackage not found"
**A**: ops-math 8.5.0 需要 CANN toolkit 8.5.0。用 `find /usr/local/Ascend -name bisheng` 确认 toolkit 已安装。

### Q: sudo 权限不足
**A**: 密码在 `~/ccl/passwd` 文件中。用 `echo '<pw>' | sudo -S <cmd>` 执行。

### Q: 第一次执行很慢
**A**: 正常。在线编译第一次需要 bisheng 编译 kernel（几秒到几十秒），后续从 `/dev/shm/op_cache` 读取缓存。
