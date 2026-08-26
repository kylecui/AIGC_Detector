# W15 Gate Review: Register-Gated Binoculars-Floor OR-Rule — DEPLOY 决策记录

**日期**: 2026-08-21
**决策**: **DEPLOY**（`enabled: true`, cutoff 0.46）——检测器获得第一个真实判别能力提升
**评审人**: 主agent（Sisyphus）；依据 plan v2.1 D4/D5 + W12 双层门协议

---

## 1. 评审过程发现并修复的缺陷（先于部署）

| # | 缺陷 | 修复 |
|---|---|---|
| 1 | 置信度语义混杂：翻转判定时取`max(旧Human置信, bino_p)`——旧"对Human的信心"为新AI判定背书，最坏0.98薄证据 | 翻转时`confidence = bino_p`（新判定的证据本身）；单测同步收紧 |
| 2 | 验证脚本首轮选例错误（案例2选了ensemble已捕获文档、案例3选了register门不命中的文档） | 改选ensemble漏检的GLM-B样本与必命中门的严正声明——**暴露的不是规则缺陷而是预期错误**，同时量化了门链的边界（见§4） |

## 2. 门控结果（全部PASS）

**W12双层门**（82篇人类探针全集，完整binoculars覆盖）：
- **L1灾难护栏**: FPR 15/82 (18.3%) → 17/82 (20.7%) [Wilson 13.4%, 30.7%]，+2.4pp << +25pp → **PASS**
- **L2配对检验**: 判定翻转2例（06-apology-yuxin / 14-chengqing-fjnu，均为预测的重叠区0.46-0.54内人类），单侧二项p=0.490 → **PASS**（非系统性）

**实时端到端验证**（真实路由链路，4/4 PASS）：
| 案例 | 结果 |
|---|---|
| FN-1（编辑后AI文本） | verdict保持Human-written，floor不触发（bino 0.343 < 0.46），置信0.89→0.59校准 ✅ |
| GLM-B合约文档（ensemble漏检，t42） | **floor触发，Human→AI**，decision_rule溯源在场，bino 0.6706，校准后置信0.53 ✅ |
| 人类严正声明（zibo） | verdict保持Human，floor不触发（bino 0.314） ✅ |
| casual帖（无caveat） | floor整链路跳过，254ms ✅ |

**回归**: 298/298测试全绿（含7个floor专项单测+部署状态锚定测试）

**延迟实测**: formal-zh且早退的文档，floor强制补跑binoculars **17.6-23.1s**（Qwen2-7B对，4-bit，12GB共享GPU）；binoculars已在breakdown中的查询<3ms

## 3. 能力变化（部署前后，探针实测）

| 场景 | 部署前漏检 | 部署后漏检 |
|---|---|---|
| GLM-4-9B 合约生成formal-zh | 76% | **~0%**（Wilson [0, 12%], n=59） |
| DeepSeek-V3.2 合约生成 | 97% | **~0%**（[0, 12%], n=30） |
| Qwen3-8B / 14B 合约生成 | 93% / 77% | **~0% / ~0%** |
| 人类formal-zh FPR | 18.3% | 20.7%（+2.4pp，区间内） |
| FN-1类（人工编辑后AI文本） | 漏检 | **仍漏检**（结构边界：floor捕获原始生成，编辑即脱靶） |

## 4. 诚实边界（随部署公开）

1. **门链乘积**：floor仅在register门命中时被咨询——词法门的formal-zh召回率约80%（62篇中约50篇命中；稀疏套语文档如简短银行公告、MEE式督察通报不命中）。门不命中=floor不保护。
2. **raw-vs-edited边界**：floor的统计检测面止于原始生成文本。任何实质性人工编辑（FN-1式）即落入人类区间——这是规范松弛框架的结构性结论，不是阈值可调的参数。
3. **重叠区人类**：0.46-0.54区间内2-4篇人类公文会被升级误判（本次实测2篇）。
4. **延迟**：formal-zh请求p50增加约20s（强制binoculars）。
5. **校准交互**：floor翻转的判定经W11温度缩放后显示为中等置信（如0.53）——正式语域内一切置信被压低是设计使然（D1）。

## 5. 复审条件

- register门召回改善（词表扩展或表征级门）→ 可重扫阈值
- binoculars模型对升级（Qwen2→更强）→ 重跑w3b_floor_analysis
- 探针集扩充（人类formal-zh >100篇）→ 收紧FPR区间后重估cutoff
