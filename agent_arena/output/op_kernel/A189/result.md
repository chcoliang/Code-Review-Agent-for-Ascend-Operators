循环中创建的aclTensor对象（self/other/out）从未调用`aclDestroyTensor`销毁（第66-69行注释处），导致tensor描述符引用计数泄漏，host内存持续增长直至耗尽。触发条件：循环多次调用aclnnMul。
