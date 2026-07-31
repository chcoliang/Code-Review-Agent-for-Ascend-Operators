`CheckInplaceMulNotNull`函数缺少对`other`参数的空指针检查（第152-155行），当`other`为nullptr时会导致后续解引用崩溃。触发条件：调用`aclnnInplaceMulGetWorkspaceSize`时传入nullptr的other tensor。
