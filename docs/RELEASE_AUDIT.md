# 最终 release audit 与实验清单

本轮为 **F5 控制实验与最终发布审计修订**。E1–E6 和首次 F5 test 已完成；项目关闭，只允许只读审计、合成测试、文档修正与 PDF 重建。冻结基线为 `cb325aae39a43c42c002744ca9c7b02510d3cb69`；原 `badecf4` 证据和全部正式产物继续保留。不训练、不推理、不生成 bootstrap 重复、不拟合校准器、不重跑 benchmark。

中文 PDF 为 **19 页**，当前 QA 见 `results/release_audit_v3/pdf_review_v3.json`。Arial Unicode TrueType 子集嵌入；逐页检查中文、表格、图例、分页、页脚与引用。独立 `.venv-report` 使用 `requirements-report.txt`。

## 公开仓库隐私与路径闭环（2026-09-27）

对 Git 已追踪文件逐行扫描，冻结修订前的历史命中为 **155 个文件、228 处**，全部位于 `results/` 下的 JSON（74 个）或日志（81 个）；它们包含机器本地的 macOS 用户目录绝对路径，会暴露本机用户名，也不能作为可直接复制的复现命令。为保持冻结证据哈希，**没有批量修改这些历史文件**，也没有重写历史。因此不能声称所有历史证据本身完全可移植。正式复现入口采用本文件和 [README](../README.md) 中的仓库相对路径，从仓库根目录运行只读检查。

[脱敏路径基线](../results/public_artifact_privacy_audit_v1/baseline.json)按文件、行号、路径类型及原行 SHA256 记录历史命中，不存放原始用户名或完整本机路径；[公开产物审计](../src/59_audit_public_artifacts.py)只放行这些精确历史行，新增 macOS/Linux 个人目录路径或疑似凭据会使检查失败。独立扫描 private key、AWS/GitHub/OpenAI key、Bearer token、命名密钥赋值和密码赋值模式，**当前未检出符合这些模式的真实凭据**；F5 的 `sequence token`、`token_count` 等研究字段不作为访问凭据。该模式扫描不能代替人工审查所有二进制内容或识别一切未知密钥格式。

磁盘现有 **59 个**同名 `test_predictions.parquet`：由冻结 registry 与 F4/F5 manifest 确认 **38 个正式 test 预测产物**；F4 posthoc manifest 中 **13 个**派生预测，加上四个校准 manifest 对应的 **4 个**校准预测，共 **17 个派生/校准/后处理产物**；另有 **4 个**位于 `_smoke` 目录的 smoke 产物。四类路径互不重叠并覆盖磁盘清单；59 个文件**不等于**59 次正式 test 推理。旧运行是否曾覆盖文件，仍不能仅靠现存文件数量证明。上述核算只读取目录、registry 与 manifest，没有加载预测、运行模型或重算指标。

本次修订只涉及发布文档、独立审计代码、合成测试和脱敏基线；中文报告正文及 19 页 PDF 未改。
在既有 65 项基础上新增 7 项路径与凭据审计夹具，当前全套为 72 项。

## 本轮审计范围与结论修订

`37_verify_frozen_delivery.py` 核对历史 F4 冻结交付；原 `38_release_audit.py` 的数值断言主要覆盖 F4 与 D2–D4。其 `passed: true` 不能解释为 E1–E6 获得相同深度的审计。新增 `58_audit_e1_e6_release.py` 独立复算 E1 namespace、E3、F5 逐 seed 汇总与六组比较、保存的区间/p/Holm 算术、donor 汇总、serving 比值、registry 与 PDF/文档声明。范围为：

> saved-result arithmetic, cross-table consistency, protocol metadata and document claims; no model inference, no resampling and no causal validation

小型表格的算术一致性与哈希不可变性是不同检查；两者都不证明实验因果语义正确。保存的 bootstrap replicates 仅用于检验已有 quantile、p 与 Holm 计算，不产生新抽样。

- **Donor**：控制是 deterministic different-user donor assignment with a measured approximately 0.08% user-level reuse deviation。Validation reuse excess **884/1,131,745（0.078109%）**；test **917/1,134,206（0.080850%）**。869/911 个 donor 被复用，最大 4/3 次；涉及 1,753/1,828 个 target，其中 575/562 个当日有曝光，覆盖 **15,389（0.475843%）/12,845（0.388223%）** 行。无 self-match、长度错配、未来事件、所有权或顺序违规。完整 8 日及分桶统计见 [donor 审计](F5_DONOR_UNIQUENESS_AUDIT.md)。
- **95%**：仅是当前不同用户 donor 控制下 matched−mismatched / matched−zero 的 AUC 描述性比率（约 94.52%），mismatched 保留约 5.5%；不是身份因果贡献。Category TV **0.04982**、年龄 **3.75 vs 4.08 天**、曝光加权活跃度 **138.23 vs 87.72** 仍不平衡。长度精确一致，粗桶近似匹配；用户级平衡不等于曝光级平衡，不能把活跃度差简单归因于 0.08% donor 复用。没有重训严格一一置换版本，无法给出反事实指标，也不能以比例小断言影响可忽略。
- **Serving**：统一称 **in-memory prepare + inference**；旧 `end_to_end_*` 字段不改。`read_batches()` 在计时前，排除 Parquet 打开、磁盘读取、解压和反序列化。DeepFM/F5 pure @8192 为 **1,530,445 / 658,608 rows/s**：F5 为 **1/2.32**；prepare+infer 为 **1/3.27**，峰值 RSS **7.48 倍**。57,345 个新增参数不代表计算可忽略；7.23 亿事件常驻主导内存，计算来自 200-token 前向及准备。未测真实磁盘/Parquet I/O 的冷缓存在线延迟。
- **GAUC**：预选 mean 对 zero/F4 的区间含零；attention−zero、attention−attention mismatched、attention−mean 的区间排除零，但仅是未经 GAUC 指标族多重校正的名义探索性证据。删除无条件的 F5 无用户内排序增益结论。
- **UNKNOWN**：高 UNKNOWN 组没有观察到更小的 AUC 增益，不识别移除 masking 的效果。**4.3029%** 是全事件库分母；test matched 保留 token 是 **3.941954%**（24,153,874 / 612,738,660）。
- **Registry**：100 目录、16 阶段、0 未分类、38 个保存的正式 test 预测产物，为冻结 registry 快照；新审计目录不回写旧 registry。组成 **15+15+3+3+2**。F4/F5 的 once-only marker 提供执行约束；仅靠旧预测文件计数不能独立证明所有历史运行都从未覆盖重跑。
- **Git**：122 个 `… 2.*` 未追踪副本全部 SHA256 相同后才移动至仓库外可恢复目录 `../final_project_duplicate_quarantine_20260927/`，清单见 `results/release_audit_v3/duplicate_quarantine.json`。修订分支 `fix/f5-audit-clarifications`，只 push 此分支，不合并 main；不重写历史、不删除已有 AI 署名、新 commit 不添加 AI trailer。

## 最终文件

- [中文 PDF](FINAL_REPORT_ZH.pdf)：19 页，包含实验清单、结论、限制、八幅已有图及仓库内引用。
- [中文文本源](FINAL_REPORT_ZH.md) 与 [英文最终报告](FINAL_REPORT.md)。
- [README](../README.md)：总览、数值、指标方向、随机种子和安全运行入口。
- 历史 v1 证据：[数值与文件清单](../results/release_audit_v1/consistency.json)、[冻结完整性](../results/release_audit_v1/frozen_integrity.json)、[全部测试日志](../results/release_audit_v1/tests.txt)、[PDF 逐页检查](../results/release_audit_v1/pdf_review.json)。

## 逐项核对与修订记录

| 核对项 | 发布处理 |
|---|---|
| README / 中英文报告 / CSV / JSON | 核对整体、逐 seed、汇总、差值、bootstrap 点估计、冷启动、历史长度、GAUC 贡献和每日计数；原始数值文件不变 |
| 数字冲突 | D2 文档已知用户 GAUC 贡献原写约 98.88%；保存值约 98.885477%，统一按四舍五入写 98.89%；仅修正文案 |
| 共同漏报 | 71.38% 始终附每模型 top-10% 曝光预算，分母为所有 test 点击；并集曝光覆盖为 14.61%，不是 10% |
| 组件删除 | 全部明确为 frozen component deletion 的 sensitivity diagnostic，不是 retrained ablation |
| 时间与重复曝光 | 明确为描述性关联；不能归因为漂移或广告疲劳的因果作用 |
| 图题与表题 | 八图逐项核对来源、分母、指标方向、缩放与横轴截断；中文图序为原图 01、02、05、04、08、03、06、07 |
| COPC | 实际点击 / 预测点击 = CTR / 平均预测；1 最佳，<1 高估、>1 低估；不能简单标成越大越好 |
| 其他指标 | AUC / PR-AUC / GAUC ↑；LogLoss / Brier / ECE ↓；缺乏正负类时 AUC 不可定义，不填零 |
| 随机性 | tabular/backbone 42；残差 42/2026/3407；错配 20260926；原 bootstrap 20260927，1,000 次；zero 只有一个确定性参照 |
| 链接与路径 | README 使用仓库相对链接，检查本地链接目标；无本机绝对路径。历史文件中保留的绝对路径不伪装为通用命令 |
| 依赖 | 核心宽范围与已测锁定版本一致；历史 sklearn 1.8 checkpoint 在当前 1.9.1 的兼容限制保留；排版依赖独立 |
| PDF | Arial Unicode TrueType 子集嵌入；逐页检查中文、表格、图例、分页、页脚与引用；没有模型或数据加载 |
| E2 序列结论适用范围 | README 与中英文报告统一收窄为"过去点击广告 category、L=20、test 平均 4.69 token、30.21% 空历史"；明确不推广到 `behavior_log` 行为序列，也不构成对 DIN 的一般否定。冻结证据文档保留原措辞，不回改 |
| E2 COPC 换算 | 0.581 对应预测比实际**高约 72%**（`1/0.581 − 1`）；与"实际比预测少 42%"（`1 − 0.581`）分母不同，两处不再混用 |
| E2 isotonic 与排序指标 | 英文文档原写 "reduces AUC" 过于绝对。isotonic 为单调非降阶梯函数，只产生并列、不产生新反转；并列可替换正确或错误排序对，故排序指标**两个方向都可能变化**，已保存结果中为下降。`ece_bins=20` 属 ECE 分箱，非 isotonic 超参（PAVA 自动定台阶）|
| E2 校准切分表述 | impression-level 80/20 **不是**校准器记住用户标签的泄漏（校准器只见概率与标签）；真实局限是同用户曝光非独立导致的样本依赖，可能低估方法选择的不确定性 |
| E2 行为窗口与 F3 首日 | 有效行为窗口为 2017-04-22 起 22 天（占全部事件 99.9786%），更早为 653 个日期共 127,136 条零散/损坏记录；训练首日 05-06 的 3,270,348 条（16.3%）F3 退化为 prior 且 `has_past_feedback=0`，**予以保留未删除** |
| E1–E6 运行环境守卫 | 扩展的第一轮误用了另一个本地解释器（polars 1.40.1 / torch 2.11.0 / sklearn 1.8.0），全部旧于 `requirements-tested.txt`（1.44.2 / 2.13.0 / 1.9.1）。冻结产物由 `results/sequence_selection_v1/environment.json` 记录的 `ctr_project` 解释器产生，异环境产物与其不可比。新增 `src/project_environment.py`：任何正式数据或模型入口在环境不符时**直接拒绝运行并逐项列出差异**；各 manifest 记录实际解释器。E1/E3 已在正确环境重做，所有报告数值不变（723,268,134 事件；覆盖率 99.8966% / 99.2807% / 99.5280%；bootstrap 区间逐位相同）。错误环境下的 F5 产物全部删除重建。行为存储 Parquet 的字节大小在两个解释器下不同而内容相同——这正是必须重建而非仅记录的原因 |
| E5 F5 结论边界 | F5 在全局 AUC 上超过 null（+0.00127246，Holm p=0.003996）与 F4（+0.00094211），约 95% 为已实现 donor 对照下的描述性 AUC 差值比率，非身份因果贡献；mean 对 zero/F4 的 **GAUC 差值区间均含零**；attention 三项次比较的 GAUC 区间排除零，仅属未经该指标族多重校正的探索性证据。attention 相对 mean 首次区间排除零（+0.00023967），但效应约为 matched−mismatched 差值的五分之一、约两个 seed 标准差，且 mean matched 仍为预先选定主模型，test 不改变该选择 |
| E5 donor 敏感性 | 排除 1,457 行 bucket relaxation 后，matched−mismatched 在 AUC 上最多变化 9.0e-7、GAUC 上 5.3e-6，符号全部保持；第 3 级放宽零行。该保存排除敏感性较小，但不覆盖 donor 复用影响。诊断性分析，不替代主分析 |
| E5 双冷启动 | F5 在 new_user+new_ad 上为 **−0.001741**（28,331 行，无区间）。只报告符号，不作解释 |
| E6 部署取舍 | F5 的内存 prepare + inference 吞吐为其 backbone 的 **1/3.27**（单行计算成本约 3.27 倍），峰值内存 **7.48 倍**，换 +0.00127 AUC。早期文案误写为「3.3 倍吞吐」，方向相反，已统一修正。内存来自 723,113,323 条事件常驻，**不是**残差的 57,345 参数；流式读取、内存映射、分层缓存或离线特征服务均可优化该实现 |
| E6 PDF 重建 | 按最终正文重新渲染中文 PDF：本轮 19 页（历史 E6 PDF 为 18 页，分页断言使用显式 `--expected-pages` 参数并记录）。渲染器新增三项修正：Markdown 行内 `**强调**` 与反引号此前被原样输出为字面字符（基线正文不含这些标记，是 E2 之后引入的），现已转换；表格加 `KeepTogether`，不再被拆到两页留下孤立表头；两列表格按实际内容宽度自适应列宽，避免「长标签 + 短数字」表被压成 10%/90% |
| E6 正式 test 次数 | 早前写作「38 次（F4 15 + F5 15）」算术不成立。遍历 manifest、once-only marker 与 registry 核实：**38 = F4 15 + F5 15 + LR 信息层 3 + 表格基准 3 + 旧命名 LR 2**，每个目录各含恰好一个 test 预测文件。组成已写入 README、中英文报告 |
| E2 公开结果对标 | 引入 DSIN (IJCAI'19) 同数据集表作为**背景参照**，明确不是直接复现：序列来源、切分、特征预算、backbone 与训练方式均不同；公开点估计不称显著 |

`src/38_release_audit.py` 主要对历史 F4/D2–D4 现有小型表格做汇总算术一致性核对，不从预测重新估计指标、不执行 resampling、不拟合校准器。CSV/JSON 清单包含当前 Git 追踪的结果文件（含历史记录与 smoke 元数据）；历史文件有自身作用域，不要求不同数据集或不同运行的数字相等。现有图表由冻结输入哈希、绘图公式及逐图视觉复核共同核对，不重新运行绘图实验。

本轮记录：[E1–E6 算术与声明审计](../results/release_audit_v3/e1_e6_audit.json)、[冻结完整性](../results/release_audit_v3/integrity_verification.json)、[65 项测试](../results/release_audit_v3/unit_tests_final.log)、[19 页 PDF QA](../results/release_audit_v3/pdf_review_v3.json)。

## 最终实验清单

| 证据单元 | 状态 / 范围 | 复现与定位入口（只读） |
|---|---|---|
| 原始数据与 F1/F2/F3 | 26,557,961 曝光；北京时间严格 past-only | `data/processed/base/`、`results/model_input_f3/` |
| LR F1 / F12 / F123 | 已训练；同优化器方案，不同 selected epoch 3/3/2 | `results/lr_f1/`、`results/lr_f1_f2/`、`results/lr_f1_f2_f3/` |
| LightGBM / Wide & Deep / DeepFM | 已冻结单 seed 42 | `results/lightgbm_f1_f2_f3/`、`results/wide_deep_f1_f2_f3/`、`results/deepfm_f1_f2_f3/` |
| F4 数据与错配审计 | L=20；正点击 category；严格早于曝光日期 | `results/f4_click_daily_v1/` |
| 序列 validation 与冻结 | 四条件 × 三残差 seed + 硬零参照 | `results/sequence_selection_v2/` |
| 正式 test | 15 checkpoint 各一次；13 个独特预测器 | `results/sequence_test_v2/`；保留 once-only marker |
| 用户 bootstrap 与校准 | 完成且冻结；不再执行 | `results/sequence_posthoc_v1/` |
| D2 共同错误与异质性 | 保存预测的描述性分组 | `results/deep_dive_d2_v1/` |
| D3 组件与信息层 | validation-only sensitivity diagnostic | `results/deep_dive_d3_v1/` |
| D4 时间与重复曝光 | 描述性关联，无 rolling-origin 重训 | `results/deep_dive_d4_v1/` |
| D5 图表与证据索引 | 八图，PNG + PDF；主张与限制映射 | `results/deep_dive_final_v1/`、`figures/deep_dive_v1/` |
| E1 跨源 namespace 审计 | 全量 723,268,134 事件；F5 准入判定通过 | `results/behavior_namespace_audit_v1/`、[文档](BEHAVIOR_NAMESPACE_AUDIT.md) |
| E3 冻结表格模型 bootstrap | 复用已保存 test 预测；1,000 次用户 cluster 重抽样 | `results/frozen_model_bootstrap_v1/`、[文档](FROZEN_MODEL_BOOTSTRAP.md) |
| E4 F5 行为序列数据 | 26,557,961 曝光；L=200；完整性全零；独立审计通过 | `results/f5_behavior_v1/` |
| E5 F5 验证与冻结 | 17 run / 85.6 分钟；lr 5e-4；15 checkpoint 冻结 | `results/f5_selection_v1/`、[文档](F5_BEHAVIOR_VALIDATION.md) |
| E5 F5 正式 test | 15 checkpoint 各一次；1,000 次 bootstrap；13 个 isotonic 校准器；7 组预注册分区；donor 敏感性 | `results/f5_sequence_test_v1/`、[文档](F5_BEHAVIOR_TEST_RESULTS.md) |
| E6 统一 serving benchmark | 6 模型；独立子进程；纯模型与内存 prepare + inference 双口径 | `results/serving_benchmark_v1/`、[文档](SERVING_BENCHMARK.md) |
| E6 实验 registry | 100 个已提交目录 / 16 阶段 / 0 未分类 / **38 次正式 test 推理**（F4 15 + F5 15 + LR 信息层 3 + 表格基准 3 + 旧命名 LR 2） | `results/experiment_registry.csv`、[索引](EXPERIMENT_INDEX.md) |

没有补做 LightGBM F1/F12、独立 retrained FM/DNN 消融、三个独立 backbone、多时期泛化检验或真实冷缓存磁盘 I/O 在线延迟实验。原提案中的缺项保持为缺项。

## 主要结论与限制

历史 F3 是本协议中最大的已观察信息增益。用户匹配序列相对错配及 zero 提供小幅、已有用户 cluster 区间支持的增量；F4 attention 未显示稳健实际优势，F5 attention 有小幅探索性差值。序列未超越强全局 LR，但 GAUC 的排序可以不同。校准整体改善仍伴随双冷启动 COPC≈0.581；长度 20 的 AUC 增益与 LogLoss 恶化并存。

这些结论受单一冻结 backbone、表格模型单 seed、两个 held-out 日期及部分 test 先前已曝光的限制。分组是在已知整体结果后、对应新分组结果前固定，不能追称原始预注册。已有校准是 impression-level 80/20，不是 user-grouped OOF。时间漂移、稀疏证据与重复曝光的因果贡献均未被识别。

## 数据目录与运行命令

所有命令从仓库根目录运行。`data/raw/` 保存原始数据；`data/processed/` 保存 base、特征、公共 F123 输入、F4 与诊断中间表；`artifacts/encoders/` 保存词表与 scaler；`artifacts/models/`、`artifacts/predictions/`、`artifacts/calibrators/` 保存大型冻结产物。以上大型文件主要为 Git ignored，需独立备份；提交 manifest 不等于备份数据或模型。`results/` 的小型 CSV/JSON 和 `figures/` 已版本化。

```bash
# 已有 CTR 环境：仅完整性、汇总核对和合成夹具测试
python src/37_verify_frozen_delivery.py
python src/38_release_audit.py
python src/58_audit_e1_e6_release.py
python src/59_audit_public_artifacts.py
python -m unittest discover -s tests -v
python -m py_compile src/38_release_audit.py src/39_render_final_report.py
git diff --check
```

全量测试包含微型合成数据的梯度、reload、bootstrap 和校准算法正确性夹具；不读取正式数据、不重跑已完成实验。`37` 的完整校验需要本地 ignored 产物；仅有 Git 仓库不能通过全部大文件哈希核验。`38` 可单独核对保存的汇总。

```bash
# 可选：独立排版环境，只重建报告副本，不执行任何实验
python -m venv .venv-report
.venv-report/bin/python -m pip install -r requirements-report.txt
.venv-report/bin/python src/39_render_final_report.py \
  --font "$CTR_REPORT_FONT" \
  --output data/release_copy/FINAL_REPORT_ZH.pdf \
  --render-dir data/release_copy/pages \
  --expected-pages 19
```

先将 `CTR_REPORT_FONT` 设置为可嵌入、覆盖全部中文字符的 TrueType 字体文件路径；本次使用系统 Arial Unicode.ttf，未将字体文件再分发。脚本检查缺字、字体嵌入与 19 页分页，拒绝覆盖已有 PDF 或渲染目录。中文源文件固定为 `docs/FINAL_REPORT_ZH.md`；命令中的新输出路径须尚不存在。PDF 页面的 PNG 为本地 QA 文件，不需要提交。

历史 v1 发布为 20 项测试；本轮为 65 项全部通过（包含 3 项新增 donor 夹具及 3 项审计算术夹具）。数值核对、冻结哈希检查、PDF 逐页复核、Python 编译及 `git diff --check` 均在本次提交前执行。历史 verification 的文档哈希描述其当时版本；本次文案修订与中文 PDF 由新的 release 清单记录，不覆盖旧 verification。
