#!/bin/bash
# deploy_verify_all.sh - 批量部署验证所有op_api层bug
# 使用方法: bash deploy_verify_all.sh
# 需要sudo权限(密码在~/ccl/passwd)

set -e
PROJ=/mnt/model/chencongliang/ccl/Code-Review-Agent-for-Ascend-Operators
SRC=$PROJ/ops-math/math/mul/op_api/aclnn_mul.cpp
LIB=/usr/local/Ascend/cann-8.5.0/lib64/libopapi_math.so
BUILD=/dev/shm/ops_final
PW=$(cat ~/ccl/passwd)

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$ASCEND_HOME_PATH/compiler/bin:$PATH
export PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export ASCEND_OP_COMPILER_CACHE_MODE=enable
export ASCEND_OP_COMPILER_CACHE_DIR=/dev/shm/op_cache
mkdir -p /dev/shm/op_cache

RESULTS_FILE=$PROJ/deploy_verify_results.txt
echo "=== 部署验证结果 $(date) ===" > $RESULTS_FILE
echo "Case|Bug|Correct|Buggy|Status" >> $RESULTS_FILE

inject_and_test() {
  local CASE=$1 DESC=$2 INJECT=$3 TEST=$4
  
  # 注入
  cp ${SRC}.orig $SRC
  eval "$INJECT"
  
  # 编译
  cd $BUILD && make opapi_math -j8 >/dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "$CASE|$DESC|N/A|COMPILE_FAIL|❌" >> $RESULTS_FILE
    echo "$CASE: COMPILE_FAIL"
    cd $PROJ; return
  fi
  
  # 部署buggy版
  echo "$PW" | sudo -S cp $BUILD/libopapi_math.so $LIB 2>/dev/null
  rm -rf /dev/shm/op_cache/*
  
  # 运行buggy
  cd $PROJ/npu_tests
  BUGGY_OUT=$(timeout 30 ./$TEST 2>&1)
  BUGGY_EXIT=$?
  if [ $BUGGY_EXIT -eq 139 ]; then BUGGY="SEGFAULT"
  elif [ $BUGGY_EXIT -eq 124 ]; then BUGGY="TIMEOUT"
  else BUGGY=$(echo "$BUGGY_OUT" | grep -v "^\[" | grep -oP "ret[= ]+\d+|code[= :]+\d+" | tail -1)
    [ -z "$BUGGY" ] && BUGGY="exit=$BUGGY_EXIT"
  fi
  
  # 恢复正确版
  cd $PROJ && cp ${SRC}.orig $SRC
  cd $BUILD && make opapi_math -j8 >/dev/null 2>&1
  echo "$PW" | sudo -S cp $BUILD/libopapi_math.so $LIB 2>/dev/null
  rm -rf /dev/shm/op_cache/*
  
  # 运行correct
  cd $PROJ/npu_tests
  CORRECT_OUT=$(timeout 30 ./$TEST 2>&1)
  CORRECT_EXIT=$?
  if [ $CORRECT_EXIT -eq 139 ]; then CORRECT="SEGFAULT"
  elif [ $CORRECT_EXIT -eq 124 ]; then CORRECT="TIMEOUT"
  else CORRECT=$(echo "$CORRECT_OUT" | grep -v "^\[" | grep -oP "ret[= ]+\d+|code[= :]+\d+" | tail -1)
    [ -z "$CORRECT" ] && CORRECT="exit=$CORRECT_EXIT"
  fi
  
  # 判定
  if [ "$BUGGY" != "$CORRECT" ]; then
    STATUS="✅"
  else
    STATUS="⚠️ SAME"
  fi
  
  echo "$CASE|$DESC|$CORRECT|$BUGGY|$STATUS" >> $RESULTS_FILE
  echo "$CASE ($DESC): correct=$CORRECT buggy=$BUGGY $STATUS"
  cd $PROJ
}

# === 开始验证 ===

# 参数校验类
inject_and_test "A01" "空指针深层" \
  "sed -i '/inline static bool CheckMulNotNull/,/^}/{s/OP_CHECK_NULL(out, return false);/(void)out;/}'; sed -i 's/CHECK_RET(CheckMulNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR)/CHECK_RET(CheckMulNotNull(self, other, out), ACLNN_SUCCESS)/' \$SRC" \
  "A01_bug1"

inject_and_test "A02" "Shape跳过" \
  "LINE=\$(grep -n 'inline static bool CheckMulShape' \$SRC|head -1|cut -d: -f1); BODY=\$((LINE+2)); sed -i \"\${BODY}i\\\\  return true;\" \$SRC" \
  "A02_bug1"

inject_and_test "A03" "DOUBLE硬拒" \
  "LINE=\$(grep -n 'auto ret = CheckMulParams' \$SRC|head -1|cut -d: -f1); sed -i \"\${LINE}a\\\\  if (self->GetDataType() == DataType::DT_DOUBLE) { return ACLNN_ERR_PARAM_INVALID; }\" \$SRC" \
  "A03_bug1"

inject_and_test "A05" "白名单删DOUBLE" \
  "sed -i '/ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST/,/};/{s/DataType::DT_DOUBLE, //}' \$SRC" \
  "A03_bug1"

inject_and_test "A06" "白名单过宽深层" \
  "LINE=\$(grep -n 'inline static aclnnStatus CheckMulParams' \$SRC|head -1|cut -d: -f1); BODY=\$((LINE+2)); sed -i \"\${BODY}i\\\\  return ACLNN_SUCCESS;\" \$SRC" \
  "A10_bug1"

inject_and_test "A07" "错误码伪装" \
  "sed -i 's/CHECK_RET(CheckMulNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR)/CHECK_RET(CheckMulNotNull(self, other, out), ACLNN_SUCCESS)/' \$SRC" \
  "A07_bug1"

inject_and_test "A15" "NonContiguous删" \
  "sed -i 's/l0op::IsMulSupportNonContiguous(self, other)/true/' \$SRC" \
  "A15_bug1"

inject_and_test "A173" "InplaceMul other" \
  "LINE=\$(grep -n 'inline static bool CheckInplaceMulNotNull' \$SRC|head -1|cut -d: -f1); OCHECK=\$(grep -n 'OP_CHECK_NULL(other' \$SRC|awk -F: -v start=\$LINE '\$1>start{print \$1; exit}'); sed -i \"\${OCHECK}s/.*/  (void)other;/\" \$SRC" \
  "A173_bug1"

echo ""
echo "=== 验证完成 ==="
cat $RESULTS_FILE
