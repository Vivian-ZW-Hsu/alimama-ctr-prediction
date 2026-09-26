# 最终 release audit 与实验清单

本次只核对已经保存的证据与交付文件。禁止新增模型、训练、test 推理、bootstrap、校准拟合或 post-hoc 分析；发布后停止项目扩展。冻结证据仍为 `badecf4` 和 [DIN_F4_TEST_RESULTS.md](DIN_F4_TEST_RESULTS.md)。

## 最终文件

- [中文 PDF](FINAL_REPORT_ZH.pdf)：12 页，包含实验清单、结论、限制、八幅已有图及仓库内引用。
- [中文文本源](FINAL_REPORT_ZH.md) 与 [英文最终报告](FINAL_REPORT.md)。
- [README](../README.md)：总览、数值、指标方向、随机种子和安全运行入口。
- [数值与文件清单](../results/release_audit_v1/consistency.json)、[冻结完整性](../results/release_audit_v1/frozen_integrity.json)、[全部测试日志](../results/release_audit_v1/tests.txt)、[PDF 逐页检查](../results/release_audit_v1/pdf_review.json)。

## 逐项核对与修订记录

| 核对项 | 发布处理 |
|---|---|
| README / 中英文报告 / CSV / JSON | 核对整体、逐 seed、汇总、差值、bootstrap 点估计、冷启动、历史长度、GAUC 贡献和每日计数；原始数值文件不变 |
| 数字冲突 | D2 文档已知用户 GAUC 贡献原写约 98.88%；保存值约 98.885477%，统一按四舍五入写 98.89%；仅修正文案 |
| 共同漏报 | 71.38% 始终附每模型 top-10% 曝光预算，分母为所有 test 点击；并集曝光覆盖为 14.61%，不是 10% |
| 旧 60% | 源码为 top-10% 分位阈值；旧预测数组缺失，无法精确复现旧 60%，不能归因于 probability≥0.5 |
| 组件删除 | 全部明确为 frozen component deletion 的 sensitivity diagnostic，不是 retrained ablation |
| 时间与重复曝光 | 明确为描述性关联；不能归因为漂移或广告疲劳的因果作用 |
| 图题与表题 | 八图逐项核对来源、分母、指标方向、缩放与横轴截断；中文图序为原图 01、02、05、04、08、03、06、07 |
| COPC | 实际点击 / 预测点击 = CTR / 平均预测；1 最佳，<1 高估、>1 低估；不能简单标成越大越好 |
| 其他指标 | AUC / PR-AUC / GAUC ↑；LogLoss / Brier / ECE ↓；缺乏正负类时 AUC 不可定义，不填零 |
| 随机性 | tabular/backbone 42；残差 42/2026/3407；错配 20260926；原 bootstrap 20260927，1,000 次；zero 只有一个确定性参照 |
| 链接与路径 | README 使用仓库相对链接，检查本地链接目标；无本机绝对路径。历史文件中保留的绝对路径不伪装为通用命令 |
| 依赖 | 核心宽范围与已测锁定版本一致；历史 sklearn 1.8 checkpoint 在当前 1.9.1 的兼容限制保留；排版依赖独立 |
| PDF | Arial Unicode TrueType 子集嵌入；逐页检查中文、表格、图例、分页、页脚与引用；没有模型或数据加载 |

`src/38_release_audit.py` 只对现有小型表格做汇总算术一致性核对，不从预测重新估计指标、不执行 resampling、不拟合校准器。CSV/JSON 清单包含当前 Git 追踪的结果文件（含历史记录与 smoke 元数据）；历史文件有自身作用域，不要求不同数据集或不同运行的数字相等。现有图表由冻结输入哈希、绘图公式及逐图视觉复核共同核对，不重新运行绘图实验。

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

没有补做 LightGBM F1/F12、独立 retrained FM/DNN 消融、三个独立 backbone、多时期泛化检验或公平独立吞吐实验。原提案中的缺项保持为缺项。

## 主要结论与限制

历史 F3 是本协议中最大的已观察信息增益。用户匹配序列相对错配及 zero 提供小幅、已有用户 cluster 区间支持的增量；attention 未显示稳健实际优势。序列未超越强全局 LR，但 GAUC 的排序可以不同。校准整体改善仍伴随双冷启动 COPC≈0.581；长度 20 的 AUC 增益与 LogLoss 恶化并存。

这些结论受单一冻结 backbone、表格模型单 seed、两个 held-out 日期及部分 test 先前已曝光的限制。分组是在已知整体结果后、对应新分组结果前固定，不能追称原始预注册。已有校准是 impression-level 80/20，不是 user-grouped OOF。时间漂移、稀疏证据与重复曝光的因果贡献均未被识别。旧预测缺失，因此旧 60% 无法精确复现。

## 数据目录与运行命令

所有命令从仓库根目录运行。`data/raw/` 保存原始数据；`data/processed/` 保存 base、特征、公共 F123 输入、F4 与诊断中间表；`artifacts/encoders/` 保存词表与 scaler；`artifacts/models/`、`artifacts/predictions/`、`artifacts/calibrators/` 保存大型冻结产物。以上大型文件主要为 Git ignored，需独立备份；提交 manifest 不等于备份数据或模型。`results/` 的小型 CSV/JSON 和 `figures/` 已版本化。

```bash
# 已有 CTR 环境：仅完整性、汇总核对和合成夹具测试
python src/37_verify_frozen_delivery.py
python src/38_release_audit.py
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
  --render-dir data/release_copy/pages
```

先将 `CTR_REPORT_FONT` 设置为可嵌入、覆盖全部中文字符的 TrueType 字体文件路径；本次使用系统 Arial Unicode.ttf，未将字体文件再分发。脚本检查缺字、字体嵌入与 12 页分页，拒绝覆盖已有 PDF 或渲染目录。中文源文件固定为 `docs/FINAL_REPORT_ZH.md`；命令中的新输出路径须尚不存在。PDF 页面的 PNG 为本地 QA 文件，不需要提交。

发布完整测试记录为 20 项全部通过；数值核对、冻结哈希检查、PDF 逐页复核、Python 编译及 `git diff --check` 均在本次提交前执行。历史 verification 的文档哈希描述其当时版本；本次文案修订与中文 PDF 由新的 release 清单记录，不覆盖旧 verification。
