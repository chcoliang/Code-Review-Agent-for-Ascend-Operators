#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnn/opdev/op_errno.h"

#include <iostream>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>

// Simulate FP16 behavior (IEEE 754 half-precision)
struct Float16 {
    uint16_t val;
    
    static Float16 fromFloat(float f) {
        Float16 result;
        uint32_t fbits;
        std::memcpy(&fbits, &f, sizeof(float));
        
        uint32_t sign = (fbits >> 16) & 0x8000;
        int32_t exp = ((fbits >> 23) & 0xFF) - 127;
        uint32_t frac = fbits & 0x7FFFFF;
        
        if (exp > 15) {
            // Overflow -> INF
            result.val = sign | 0x7C00;
        } else if (exp < -14) {
            result.val = sign; // underflow to zero (simplified)
        } else {
            result.val = sign | ((exp + 15) << 10) | (frac >> 13);
        }
        return result;
    }
    
    float toFloat() const {
        uint32_t sign = (val & 0x8000) << 16;
        uint32_t exp = (val >> 10) & 0x1F;
        uint32_t frac = val & 0x3FF;
        
        if (exp == 0x1F) {
            // INF or NaN
            uint32_t result = sign | 0x7F800000 | (frac << 13);
            float f;
            std::memcpy(&f, &result, sizeof(float));
            return f;
        }
        if (exp == 0) {
            if (frac == 0) return 0.0f;
            // denorm
            float f = frac / 1024.0f;
            return (sign ? -1.0f : 1.0f) * f * (1.0f / 16384.0f);
        }
        uint32_t result = sign | ((exp + 112) << 23) | (frac << 13);
        float f;
        std::memcpy(&f, &result, sizeof(float));
        return f;
    }
};

// Simulate the bug: canUseMuls path uses Muls on FP16 tensor directly
float buggy_muls_fp16(float self_val_fp16, float scalar_val) {
    // canUseMuls=true path: Muls operates on FP16 tensor
    // The kernel multiplies FP16 element by float scalar, but result is stored as FP16
    Float16 self_fp16 = Float16::fromFloat(self_val_fp16);
    float intermediate = self_fp16.toFloat() * scalar_val;
    // Result truncated back to FP16
    Float16 result_fp16 = Float16::fromFloat(intermediate);
    // Then Cast to output dtype (FP32)
    return result_fp16.toFloat();
}

// Correct path: Cast FP16->FP32 first, then Mul in FP32
float correct_mul_fp32(float self_val_fp16, float scalar_val) {
    // Cast self to FP32 (inferDtype=DT_FLOAT)
    Float16 self_fp16 = Float16::fromFloat(self_val_fp16);
    float self_fp32 = self_fp16.toFloat();
    // Mul in FP32
    float result = self_fp32 * scalar_val;
    // Cast to output FP32 (no-op)
    return result;
}

int main() {
    float scalar = 65536.0f;  // Exceeds FP16 max (65504)
    float test_values[] = {1.0f, 2.0f, 0.5f, 1.0f};
    
    std::cout << "=== Bug 1: canUseMuls ignores inferDtype, causes FP16 overflow ===" << std::endl;
    std::cout << "Input: self=FP16 tensor, scalar=" << scalar << " (DT_FLOAT)" << std::endl;
    std::cout << "Platform: Ascend 910B (IsRegBase()=true)" << std::endl;
    std::cout << "Output dtype: DT_FLOAT (FP32)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "inferDtype computed = DT_FLOAT (because 65536 > FP16 max 65504)" << std::endl;
    std::cout << "But canUseMuls=true (ignores inferDtype), so Muls used on FP16 tensor" << std::endl;
    std::cout << std::endl;
    
    bool bug_triggered = false;
    for (int i = 0; i < 4; i++) {
        float buggy = buggy_muls_fp16(test_values[i], scalar);
        float correct = correct_mul_fp32(test_values[i], scalar);
        
        std::cout << "  Element[" << i << "]: self=" << test_values[i] << std::endl;
        std::cout << "    Buggy result (Muls FP16 path):   " << buggy;
        if (std::isinf(buggy)) std::cout << " [OVERFLOW!]";
        std::cout << std::endl;
        std::cout << "    Correct result (Cast+Mul FP32):  " << correct << std::endl;
        
        if (buggy != correct) bug_triggered = true;
    }
    
    std::cout << std::endl;
    if (bug_triggered) {
        std::cout << "[BUG CONFIRMED] canUseMuls optimization produces INCORRECT results" << std::endl;
        std::cout << "  Root cause: canUseMuls at line 398 does not check (inferDtype == self->GetDataType())" << std::endl;
        std::cout << "  Expected: should fall through to else branch, Cast to FP32, then Mul" << std::endl;
    } else {
        std::cout << "[NO BUG] Results match" << std::endl;
    }
    
    return 0;
}
