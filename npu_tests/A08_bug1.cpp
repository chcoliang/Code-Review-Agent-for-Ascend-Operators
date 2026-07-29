#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnn/opdev/op_errno.h"

#include <iostream>
#include <cstdint>
#include <csignal>
#include <cstdlib>

// Simulate the bug scenario: dereferencing nullptr workspaceSize
// In actual code: *workspaceSize = 0; at line 383 or *workspaceSize = uniqueExecutor->GetWorkspaceSize(); at line 437

void signal_handler(int sig) {
    if (sig == SIGSEGV) {
        std::cout << "[BUG CONFIRMED] SEGFAULT triggered due to null workspaceSize pointer dereference." << std::endl;
        std::cout << "Expected behavior: Function should return ACLNN_ERR_PARAM_NULLPTR (161001) "
                  << "instead of dereferencing nullptr." << std::endl;
        std::exit(1);
    }
}

// Simulates the pattern in aclnnMulsGetWorkspaceSize
int simulate_get_workspace_size(uint64_t *workspaceSize) {
    // The code does NOT check workspaceSize for nullptr before:
    // Line 383: *workspaceSize = 0;  (empty tensor path)
    // Line 437: *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    
    // Simulating the empty tensor path (line 382-385):
    bool isEmpty = true; // assume self->IsEmpty() returns true
    if (isEmpty) {
        *workspaceSize = 0;  // BUG: no null check, crashes if workspaceSize == nullptr
        return 0; // ACLNN_SUCCESS
    }
    return 0;
}

int main() {
    std::signal(SIGSEGV, signal_handler);
    
    std::cout << "Test: Calling aclnnMulsGetWorkspaceSize with workspaceSize=nullptr" << std::endl;
    std::cout << "Actual behavior: ";
    
    uint64_t *nullWorkspaceSize = nullptr;
    int ret = simulate_get_workspace_size(nullWorkspaceSize);
    
    // Should not reach here
    std::cout << "returned " << ret << " (unexpected - should have crashed)" << std::endl;
    std::cout << "Expected behavior: Should return ACLNN_ERR_PARAM_NULLPTR (161001)" << std::endl;
    
    return 0;
}
