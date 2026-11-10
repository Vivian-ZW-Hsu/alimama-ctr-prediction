> **Archived working note.** Internal agent instructions / planning from the
> September 2026 rebuild, kept for transparency. It may describe states that are
> no longer current; see the [README](../../README.md) and
> [final report](../FINAL_REPORT.md) for the results.

# CTR 项目 Codex 总交接与执行章程

更新时间：2026-09-26

项目：`Click-Through_Rate_Prediction_on_E-Commerce_Advertising_Data`

本地项目根目录：以当前 VS Code workspace 为准，不依赖本文中的绝对路径。

## 1. 你的角色

你是该项目的主要工程与实验执行代理。你的职责不是只回答问题，而是在当前仓库中持续完成：

1. 审计已有代码、数据产物和实验结果；
2. 制定下一阶段的可检验计划；
3. 实现代码并加入正确性检查；
4. 先运行 smoke test，再运行正式实验；
5. 使用冻结协议进行统一评估；
6. 保存 manifest、日志、指标和预测；
7. 更新文档；
8. 在验证通过后创建清晰的 Git commit，并在允许时推送到 `origin/main`；
9. 最终形成可复现、可用于课程项目与简历展示的报告。

在每个阶段开始前，先读取仓库当前状态。仓库中的代码和结果文件是事实来源；本文提供项目意图、已知结论和执行约束。如果本文中的数字与仓库不一致，停止使用冲突数字，核对生成链路并记录差异。

## 2. 总目标与研究问题

项目使用淘宝展示广告数据，重做一个严格避免时间泄漏的 CTR 预测实验，比较：

- Logistic Regression；
- LightGBM；
- Wide & Deep；
- DeepFM；
- DIN 与受控序列基线。

核心问题不是“复杂模型是否一定更强”，而是分层回答：

1. 静态与上下文信息 F1 能提供多少信息？
2. 严格 past-only 的用户行为统计 F2 能增加多少？
3. 严格 past-only 的点击反馈统计 F3 能增加多少？
4. 在相同 F1+F2+F3 输入下，非线性与显式交互模型是否优于线性模型？
5. 用户历史序列 F4 是否提供聚合统计之外的增量信息？
6. 若有增益，来自用户身份相关的真实序列，还是仅来自历史长度、热门度等旁路信号？
7. 各模型的概率校准、用户内排序和冷启动表现如何？

## 3. 不可违反的实验原则

### 3.1 时间与泄漏

- 所有特征必须满足 point-in-time correctness。
- 训练集：北京时间 2017-05-06 至 2017-05-11。
- 验证集：2017-05-12。
- 测试集：2017-05-13。
- 验证与测试的日期不可调整。
- F2/F3/F4 的历史必须严格早于当前曝光允许的时间边界。
- 不得用未来日期的行为、标签、统计量、词表频次或 scaler 信息。
- 类别词表和数值 scaler 只从训练集拟合；固定枚举字段可使用预定义合法范围。
- 测试日 F3 可以包含验证日已经结束的历史点击反馈，因为线上逐日更新也可获得该信息；必须在 manifest 中明确记录。

### 3.2 验证集与测试集

- 所有模型结构、超参数、epoch、序列长度、校准方法和 fallback 规则只能依据训练集与验证集决定。
- 测试集只用于冻结配置后的最终报告和预先规定的诊断。
- 不得根据测试 AUC、LogLoss、冷启动结果或分桶结果回头调整模型。
- 如果发现代码正确性 bug，可以修复并重跑；必须保存修复前证据、解释为何属于正确性修复而非 test-driven tuning。

### 3.3 公平比较

- 比较模型结构时，必须保持输入信息层一致。
- DIN/F4 因增加序列输入，必须作为“序列机制实验”单独呈现，不能与只使用 F1+F2+F3 的模型混成纯架构公平比较。
- 所有正式模型使用同一 train/validation/test 切分和统一 evaluator。
- 主要排序指标：AUC、GAUC、PR-AUC。
- 概率指标：LogLoss、COPC、ECE、可靠性分箱。
- GAUC 必须报告合格用户数和覆盖曝光比例。
- 校准只改变概率解释，不应被表述为排序能力提升。

### 3.4 产物与可复现性

- 每次正式运行必须有唯一、语义明确的 `run_name`。
- 保存配置、seed、依赖版本、输入路径、数据规模、训练历史、最佳 epoch、运行时间、指标和预测路径。
- smoke 结果不得作为正式证据。
- 不覆盖已有正式结果；若修复代码后需要复用正式 run name，先把关键旧结果存入诊断目录并在 manifest 中说明。
- 大型模型、预测和中间数据可以保持 Git ignored，但 CSV/JSON manifest、汇总结果、代码和文档应提交。
- 每次提交前运行最小相关测试、`python -m py_compile`、`git diff --check`，并检查 `git status`。
- 禁止删除原始数据或大范围清理目录。禁止使用 `git reset --hard`、`git checkout --` 等破坏性命令。

## 4. 已完成的数据管线

原始数据规模约为：

- `ad_feature`：30 MB；
- `behavior_log`：22 GB；
- `raw_sample`：1 GB；
- `user_profile`：1.1 GB。

有效北京时间范围为 2017-04-22 至 2017-05-13。当前切分：

| Split | 北京日期 | 曝光数 | CTR |
|---|---|---:|---:|
| Train | 2017-05-06 至 05-11 | 20,015,245 | 0.051967 |
| Validation | 2017-05-12 | 3,234,051 | 0.049299 |
| Test | 2017-05-13 | 3,308,665 | 0.050319 |

总计 26,557,961 条曝光。

现有脚本及职责：

- `01_validate_raw.py`：原始 schema、标签、主键、时间和异常检查；
- `02_profile_raw.py`：缺失值、价格、用户属性和行为日期分析；
- `03_build_base_tables.py`：以 raw ID 连接表、北京时间转换和日期切分；
- `04_build_daily_behavior.py`：聚合 user-day 行为；
- `05_build_behavior_features.py`：构建曝光日前 1/3/7/14 天行为统计、recency 和历史标记；
- `05b_check_behavior_identity.py`：抽样核验最终特征与同用户 lookup 的一致性；
- `06_profile_model_features.py`：字段角色、类别覆盖、未知值和数值分布；
- `07_build_model_inputs.py`：训练集词表、缺失处理和公共模型输入；
- `11_build_past_ctr_features.py`：构建严格早于曝光日期的 F3 历史点击反馈；
- `12_build_f3_model_inputs.py`：将 F3 加入公共输入并统计历史证据量。

当前 `base`、`features`、`model_input`、`past_ctr`、`model_input_f3` 的三个 split 产物均存在。不要无理由重建 22 GB 行为聚合；先复用并验证已有产物。

## 5. 信息层定义

### F1：静态与上下文

包含用户、广告与曝光上下文。正式模型统一排除 `day_of_week_idx`，避免在单日 validation/test 中形成不可泛化的日期代理。编码表可能仍保留该字段，不代表正式模型使用它。

### F2：past-only 行为统计

只使用曝光日前已经结束日期的行为，包含 1/3/7/14 天窗口统计、recency 与历史存在标记。

### F3：past-only 点击反馈统计

覆盖 user、ad、category、campaign、customer、brand；平滑强度 100，首日默认 prior 0.05。增加 13 个数值特征和 6 个二元特征。必须保留其逐日滚动语义。

### F4：用户兴趣序列

尚未实现。必须先明确序列事件来源、元素语义和时间边界，再编码。候选来源包括：

1. `behavior_log` 中的用户行为事件；
2. 过去广告曝光中 `clk=1` 的正点击序列。

不得直接假设二者等价。应根据 DIN 的目标匹配语义、数据覆盖、可用 item ID、时间粒度和泄漏风险进行选择。优先选择能表示“用户对广告/商品的历史兴趣”，且能与当前候选广告的 category/ad 表示对齐的方案。

## 6. 已完成的模型与正式结果

### 6.1 LR 信息层消融

| 输入 | Test AUC | Test LogLoss |
|---|---:|---:|
| F1 | 0.595495 | 0.196781 |
| F1+F2 | 0.601790 | 0.196376 |
| F1+F2+F3 | 0.652489 | 0.192429 |

当前最强的信息增益来自 F3；F2 有小幅增益。这说明严格 past-only 的历史反馈质量比单纯增加模型复杂度更重要。

LR 正式配置：`alpha=1e-7`、`eta0=0.005`、seed 42，最佳 epoch 2。选择规则实际为：在距离最佳 validation LogLoss 不超过 0.0002 的候选中选最高 AUC。该规则与旧实验协议文本存在差异，最终必须透明记录或修订协议，不能伪装成完全预注册一致。

### 6.2 F1+F2+F3 四模型公平比较

| 模型 | Val AUC | Val LogLoss | Test AUC | Test LogLoss | Test GAUC |
|---|---:|---:|---:|---:|---:|
| LR | 0.655163 | 0.189469 | **0.652489** | **0.192429** | 0.569939 |
| LightGBM | 0.654303 | **0.189288** | 0.651318 | 0.192502 | 0.569889 |
| Wide & Deep | 0.650552 | 0.189804 | 0.646219 | 0.193161 | 0.569680 |
| DeepFM | 0.650315 | 0.189813 | 0.645933 | 0.193176 | 0.569902 |

当前可说：单次正式 test 中 LR 排名第一，LightGBM 接近；Wide & Deep 和 DeepFM 没有超过强基线。不能在完成置信区间与多种子之前声称差异显著。

LightGBM 正式配置：学习率 0.03、63 leaves、`min_data_in_leaf=2000`、最佳迭代 251，以 validation LogLoss 早停。

Wide & Deep 与 DeepFM 正式配置：embedding 8、MLP `[128, 64]`、dropout 0.3、学习率 0.0005、weight decay `1e-5`、batch 8192、seed 42、MPS。两者均在 epoch 1 最佳，训练到 epoch 3 后早停；训练损失下降而 validation 恶化，表现出明显过拟合。

## 7. 已知工程审计：UNKNOWN embedding

神经模型曾把训练时不会出现的 `UNKNOWN=1` embedding 随机初始化，导致验证/测试的新用户和新广告使用未经训练的随机向量。现已修复为：

- 所有类别 embedding 的 UNKNOWN 行固定为零；
- 一阶权重的 UNKNOWN 行固定为零；
- 初始化后及每个 epoch 后执行断言，确保保持严格为零。

修复前诊断保存在 `results/nn_unknown_embedding_diagnostic/`。修复后正式结果如上。修复改善了神经模型表现与冷启动稳定性，但不是神经模型落后 LR 的主要原因。

## 8. 统一评估、GAUC 与校准

`10_evaluate_predictions.py` 负责验证预测 ID、概率范围、标签一致性，并输出：AUC、LogLoss、PR-AUC、COPC、ECE、可靠性分箱、GAUC、冷启动四象限。

GAUC 按用户曝光数加权，只纳入同时具有正负样本的用户。测试集当前有 104,884 个合格用户，覆盖 50.8437% 的测试曝光。报告 GAUC 时必须同时报告这两个覆盖数字。

`14_calibrate_predictions.py` 的实际流程：

1. 按 `impression_id` 哈希将 validation 确定性拆为约 80% calibration fit、20% method selection；
2. 比较 Platt 与 isotonic；
3. 以 selection LogLoss 选择方法；
4. 在完整 validation 上重拟合所选校准器；
5. 应用于冻结 test。

旧协议曾写“用户分组五折 OOF 校准选择”，与当前实现不一致。不要悄悄改写历史。应在最终报告中把当前做法标记为实际协议与 deviation；若要补做 grouped OOF，只能使用 validation 选择，不得因 test 结果决定是否保留。

| 模型 | 方法 | Test LogLoss：原始 → 校准 | COPC：原始 → 校准 |
|---|---|---|---|
| LR | Isotonic | 0.192429 → 0.192346 | 1.051337 → 0.958858 |
| LightGBM | Platt | 0.192502 → 0.192507 | 1.035983 → 1.041490 |
| Wide & Deep | Isotonic | 0.193161 → 0.192849 | 0.940161 → 0.986470 |
| DeepFM | Isotonic | 0.193176 → 0.192876 | 0.939656 → 0.985826 |

校准没有保证跨日 test 改善：LightGBM 略差；两个神经模型整体概率偏差明显减小。Isotonic 造成轻微 AUC 下降可来自并列预测增加，不能解释成排序提升。

## 9. 冷启动结论

“新用户/新广告”按是否出现在训练集 raw ID 词表定义，不等于完全没有 F2/F3 历史。神经模型校准后的测试结果：

| 分组 | 曝光数 | Wide & Deep 预测偏差 | DeepFM 预测偏差 |
|---|---:|---:|---:|
| 已知用户、已知广告 | 2,659,898 | -4.06% | -4.00% |
| 已知用户、新广告 | 261,758 | +40.01% | +41.16% |
| 新用户、已知广告 | 358,678 | +18.24% | +17.86% |
| 新用户、新广告 | 28,331 | +69.21% | +70.95% |

偏差定义为 `mean_prediction / actual_CTR - 1`。正式结论：全局 calibration 能显著改善整体概率偏差，但不能消除 segment-specific cold-start miscalibration。未来可研究分组校准或 fallback，但不得根据 test 分组表现直接设计并宣称无偏的 test 改进。

## 10. 当前缺项与优先级

已有仓库审计发现：

- LightGBM 缺少 F1、F1+F2 信息层对照；
- deep-only、FM-only、DNN-only 等架构组件消融未完成，需先核对协议中的确切定义；
- 神经网络只有 seed 42，缺少三种子汇总；
- 缺少用户聚类 paired bootstrap；
- DIN/F4 尚未实现；
- 用户活跃度、广告频次、历史长度等稳健性分桶不完整；
- 缺统一推理吞吐和内存基准；
- README、requirements 与当前状态不一致；
- `requirements.txt` 缺 `lightgbm`、`torch`，且 PyArrow 版本约束与实际环境不一致；
- 早期 manifest 存在 `processed_v2` / `artifacts_v2` 历史路径残留。

优先级如下：

1. **DIN/F4 数据语义与防泄漏设计**；
2. F4 数据构建、identity audit、smoke；
3. 受控序列实验；
4. 将 DIN 接入统一 evaluator、校准和冷启动/历史长度诊断；
5. 对最终保留模型做多种子与 paired user bootstrap；
6. 补协议明确要求且能回答独立研究问题的消融；
7. 稳健性、效率、README、requirements 和最终报告收尾。

不要为了勾选清单而立即跑所有昂贵实验。先判断每个缺项回答什么问题，并复用中间产物。

## 11. DIN/F4 必须执行的受控实验

至少包含以下四个可比较条件：

1. **无序列基线**：现有 DeepFM F1+F2+F3；
2. **正确用户序列 + mean pooling**：检验序列内容整体是否提供增量信息；
3. **正确用户序列 + target attention（DIN）**：检验候选相关注意力是否优于简单聚合；
4. **错配用户序列负对照**：保持历史长度等边际分布，打破“该历史属于该用户”的对应关系。

错配负对照必须：

- 确保来源用户不等于目标用户；
- 尽量保持历史长度分布；
- 不引入未来事件；
- 使用固定 seed，可完全复现；
- 不改变 F1+F2+F3 的非序列输入；
- 明确它用于排除历史长度、流行度或额外参数量导致的伪增益。

mean pooling 与 attention 必须共享：序列来源、截断规则、embedding、非序列塔、优化器搜索空间和数据切分。除了聚合机制，不得同时改变其他关键组件。

若 F4 来自过去正点击曝光，应优先采用逐事件严格时间边界，或保守采用已完成日期边界；二者不可混用而不说明。若采用 behavior log，应说明 `btag` 事件与当前广告候选 item/category 的可对齐程度。序列元素可考虑 category 为主、adgroup 为辅，但必须先根据覆盖率和词表规模验证。

## 12. 下一阶段执行流程

### 阶段 A：只读设计审计

先完成并写入 `docs/DIN_F4_DESIGN.md`：

- 候选序列来源及语义比较；
- 最终推荐来源；
- event schema；
- 每条曝光的时间 cutoff；
- train/validation/test 历史累积规则；
- 序列元素与候选广告的对齐方式；
- 空历史、padding、UNKNOWN、max length；
- 错配负对照算法；
- 内存、磁盘和时间估算；
- 验证集内的超参数选择方案；
- 预期新增文件。

在设计无法由数据 schema 支持时，不要强行编码；指出阻塞点和最小可行替代方案。

### 阶段 B：F4 数据构建与审计

实现独立的数据构建脚本，避免污染已有 F1/F2/F3 产物。要求：

- 幂等；
- 支持 `--smoke-test`；
- 支持断点或按日期构建；
- 输出 schema、行数、序列长度分布、空历史率、UNKNOWN 率、截断率；
- 对随机抽样曝光回查原始事件；
- 断言所有序列事件早于 cutoff；
- 断言 impression ID 与标签不变；
- 记录输入文件指纹或 manifest 信息。

先运行 smoke 和 identity audit。只有全部通过才运行全量构建。

### 阶段 C：模型实现

优先复用 `16_train_tabular_nn.py` 的数据加载、数值 scaler、类别 embedding、UNKNOWN 零向量、早停、manifest 和预测格式。不要复制出相互漂移的整套训练基础设施。

新增序列模块时应支持至少：

- `pooling=mean`；
- `pooling=attention`；
- `sequence_source=matched`；
- `sequence_source=mismatched`；
- padding mask；
- 空历史的中性零表示；
- UNKNOWN 序列 token 的中性处理；
- 可复现 seed；
- smoke 模式。

先用小样本检查 shape、mask、梯度、padding 不敏感性和错配恒等条件，再运行全量验证集调参。

### 阶段 D：冻结配置与正式评估

- 只根据 validation 冻结配置；
- 首次正式 test 后不得继续调参；
- 运行统一 evaluator；
- 比较整体、GAUC、校准、冷启动、空历史与历史长度分桶；
- 保存与现有模型一致的结果结构。

### 阶段 E：统计与收尾

- 对随机神经模型至少运行 3 个预先固定 seed；
- 汇报 mean ± std；
- 使用以用户为 cluster 的 paired bootstrap 比较关键模型差值；
- 补齐协议中真正必要的信息层和架构消融；
- 更新 README、requirements、实验协议 deviation、结果表与最终报告；
- 记录推理吞吐、峰值内存和硬件环境。

## 13. 自主工作规则

你可以自主完成只读检查、代码实现、smoke、单元/一致性测试、结果解析、文档更新和非破坏性 Git 操作。遵循以下节奏：

1. 开始前运行 `git status --short` 并阅读相关代码；
2. 写出简短执行计划；
3. 每次只推进一个可验证阶段；
4. 修改后运行最小测试；
5. smoke 通过后再启动全量任务；
6. 长任务使用可恢复日志，必要时使用 `caffeinate -i` 防止 Mac 睡眠；
7. 每个阶段结束时汇报：做了什么、验证证据、关键结果、剩余风险、下一步；
8. 通过检查后创建语义清晰的 commit；
9. 推送前确认没有意外加入原始数据、大模型或预测文件；
10. 不得把本地存在但被 `.gitignore` 忽略的文件误称为已提交或已备份。

若需要作出会改变研究问题、数据口径、冻结 test 使用方式或大量计算预算的重大决定，先暂停并向用户说明选项。普通实现细节由你自主选择并记录。

## 14. 文档真实性要求

最终项目叙事必须区分：

- 已由结果直接支持的事实；
- 基于结果的解释或推断；
- 尚未验证的假设；
- 与原实验协议的 deviation；
- 工程正确性修复与模型调参。

当前可以写入结论：

1. 严格 past-only F3 是现阶段最大的性能增益来源；
2. 在统一 F1+F2+F3 输入和当前调参预算下，复杂表格模型没有自动超过 LR；
3. 四模型 GAUC 非常接近，LR 的全局 AUC 优势未转化为明显更强的用户内排序；
4. 全局 calibration 改善整体偏差，但没有解决冷启动分组高估；
5. UNKNOWN embedding 零向量是必要的正确性修复，但不是神经模型落后的主要原因。

当前不能写成确定结论：

- LR 在总体上具有统计显著优势；
- DIN 一定会优于非序列模型；
- attention 一定优于 mean pooling；
- 冷启动已经被 calibration 解决；
- 所有协议要求都已完成。

## 15. 启动指令

读取本文后，执行以下动作：

1. 对照当前仓库重新验证本文中的文件、结果和 Git 状态；
2. 不重跑已有模型；
3. 从 DIN/F4 阶段 A 的只读设计审计开始；
4. 生成 `docs/DIN_F4_DESIGN.md` 草案；
5. 在真正写 F4 数据代码前，向用户报告推荐序列来源、时间 cutoff、负对照构造和预计资源；
6. 获得方向确认后，持续推进后续实现、验证、实验、评估和提交。

项目的首要标准是：正确、可复现、可解释。模型数量和复杂度排在其后。
