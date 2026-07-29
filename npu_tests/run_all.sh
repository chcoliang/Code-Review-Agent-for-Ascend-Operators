#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/lib64:/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
for t in A0*_bug1; do
  [ -x "./$t" ] || continue
  echo "=== $t ==="
  timeout 60 ./"$t" 2>&1
  code=$?
  if [ $code -eq 139 ]; then echo "[SEGFAULT]"
  elif [ $code -eq 124 ]; then echo "[TIMEOUT]"
  elif [ $code -ne 0 ]; then echo "[EXIT=$code]"
  else echo "[OK exit=0]"
  fi
  echo ""
done
