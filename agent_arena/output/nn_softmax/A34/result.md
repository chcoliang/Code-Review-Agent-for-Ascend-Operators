# Ascend NPU 算子代码审查报告 - aclnnSoftmax (A34)

## Bug 列表

### Bug 1: 空tensor提前返回导致dim参数未校验

- **位置**: 第93-95行 (`CheckParams`函数)
- **类型**: 逻辑错误 / 参数校验缺失
- **严重程度**: 中
- **描述**: 当`self->IsEmpty()`为true时，函数直接返回`ACLNN_SUCCESS`，跳过了后续对`dim`参数的合法性检查(`CheckDim`)。这与PyTorch的行为不一致——PyTorch即使对空tensor也会校验dim是否在合法范围内，传入非法dim应报错而非静默成功。
- **触发条件**: 传入一个空tensor和一个超出维度范围的dim值（如shape为[0,3]的tensor传入dim=5）。
- **测试方案**: 构造shape为[0, 3]的空tensor，dim设为10，期望返回`ACLNN_ERR_PARAM_INVALID`，实际返回`ACLNN_SUCCESS`。

### Bug 2: 未对workspaceSize和executor指针参数进行空指针检查

- **位置**: 第109行 (`aclnnSoftmaxGetWorkspaceSize`函数入口)
- **类型**: 空指针解引用风险
- **严重程度**: 高
- **描述**: 函数参数`workspaceSize`和`executor`为用户传入的指针，但函数内部未做空指针检查即在第141-142行直接解引用(`*workspaceSize = ...`和`uniqueExecutor.ReleaseTo(executor)`)。若用户传入nullptr将导致段错误崩溃。
- **触发条件**: 调用`aclnnSoftmaxGetWorkspaceSize`时传入`workspaceSize=nullptr`或`executor=nullptr`。
- **测试方案**: 分别传入nullptr作为workspaceSize和executor参数，验证是否安全返回错误码而非崩溃。

### Bug 3: 注释与实际调用不符（调用SoftmaxV2却注释为SoftmaxGrad）

- **位置**: 第132行
- **类型**: 注释错误
- **严重程度**: 低
- **描述**: 注释写"调用SoftmaxGrad算子kernel"，但实际代码调用的是`l0op::SoftmaxV2`（前向softmax算子）。这是明显的复制粘贴错误，可能误导后续维护者。
- **触发条件**: 代码审查/维护时产生误解。
- **测试方案**: 代码走读确认，无需运行时测试。

### Bug 4: 缺少self与out之间的数据类型一致性校验

- **位置**: 第58-63行 (`CheckDtypeValid`函数)
- **类型**: 参数校验缺失
- **严重程度**: 中
- **描述**: `CheckDtypeValid`仅分别检查self和out的dtype是否在支持列表中，但未校验self和out的dtype是否一致（或是否为合法的类型转换组合）。Softmax的输出dtype应与输入一致，若self为FP16而out为FP32，当前代码不会报错，但后续`ViewCopy`可能产生未定义行为或精度问题。
- **触发条件**: 传入self为FP16类型，out为FP32类型的tensor。
- **测试方案**: 构造dtype不同的self(FP16)和out(FP32)，检查是否正确报错或结果是否正确。

### Bug 5: 空tensor场景下aclnnSoftmaxGetWorkspaceSize中executor和workspaceSize的空指针风险

- **位置**: 第121-126行
- **类型**: 空指针解引用风险
- **严重程度**: 高
- **描述**: 空tensor分支中，第123行`*workspaceSize = 0`和第124行`uniqueExecutor.ReleaseTo(executor)`同样未检查`workspaceSize`和`executor`是否为空指针。与Bug 2属于同一类问题但在不同执行路径上。
- **触发条件**: 传入空tensor且`workspaceSize=nullptr`或`executor=nullptr`。
- **测试方案**: 构造空tensor输入，传入nullptr参数，验证是否崩溃。

### Bug 6: CheckShape未对out进行维度上限检查

- **位置**: 第80-85行 (`CheckShape`函数)
- **类型**: 校验不完整
- **严重程度**: 低
- **描述**: `OP_CHECK_MAX_DIM`仅对self进行了维度上限(AXIS_LIMIT=8)检查，未对out检查。虽然第83行`OP_CHECK_SHAPE_NOT_EQUAL`会确保shape一致，但在某些异常情况下(如out的shape与self相同但存储维度不同)，可能遗漏对out的维度检查。
- **触发条件**: 传入self为7维tensor，out为等价shape但内部表示维度超过8维的tensor。
- **测试方案**: 构造维度数超过8的out tensor（shape与self一致但有额外的size-1维度），验证是否被正确拦截。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第93-95行 | 逻辑错误 | 中 | 空tensor提前返回跳过dim校验 |
| 2 | 第109行(函数入口) | 空指针解引用 | 高 | 未检查workspaceSize/executor空指针 |
| 3 | 第132行 | 注释错误 | 低 | 注释写SoftmaxGrad实际调用SoftmaxV2 |
| 4 | 第58-63行 | 校验缺失 | 中 | 未校验self与out的dtype一致性 |
| 5 | 第121-126行 | 空指针解引用 | 高 | 空tensor路径未检查指针参数 |
| 6 | 第80-85行 | 校验不完整 | 低 | 未对out进行维度上限检查 |
