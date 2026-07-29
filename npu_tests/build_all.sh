#!/bin/bash
CANN=/usr/local/Ascend/cann-8.5.0
PASS=0; FAIL=0
for f in A0*_bug1.cpp; do
  name="${f%.cpp}"
  echo "Building $name ..."
  if g++ -std=c++17 -o "$name" "$f" \
    -I${CANN}/include -L${CANN}/lib64 -L${CANN}/aarch64-linux/lib64 \
    -lascendcl -lnnopbase -lopapi \
    -Wl,-rpath,${CANN}/lib64:${CANN}/aarch64-linux/lib64 \
    -Wl,--allow-shlib-undefined 2>&1; then
    echo "  [OK]"; PASS=$((PASS+1))
  else
    echo "  [FAIL]"; FAIL=$((FAIL+1))
  fi
done
echo "Build: $PASS ok, $FAIL failed"
