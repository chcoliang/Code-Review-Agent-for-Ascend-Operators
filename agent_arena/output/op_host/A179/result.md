`GetPlatformInfo`中将平台获取的UB大小除以2（第225行`ubSizePlatform / 2`），导致tiling只使用一半UB空间，严重降低性能且可能导致tiling与kernel实际可用内存不匹配。触发条件：通过platformInfo获取UB大小的所有场景（非编译期指定）。
