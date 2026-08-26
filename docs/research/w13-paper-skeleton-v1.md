# W13 论文骨架 v1：规范松弛与AI文本检测器的胜任域

**日期**: 2026-08-18（W4c数据当日成稿；写作执行用fat-slim-writer，本件为结构契约）
**目标venue**: workshop基线（ACL/EMNLP workshop或AI safety类）+ main track stretch
**作者披露立场**: 单人+AI协作；双项目同作者的非独立性主动披露（v3.1重构7）

---

## 标题候选

1. *Specification Slack: How Output Contracts Move LLM Text Into Human Territory — and Shrink AI-Text Detectors to a Single Competence Cell*
2. *The Contract-Detector Duality: Contract Completeness as the Binding Constraint on Statistical Detectability*（对偶主打版）
3. 中文工作题：《规范松弛：输出合约、语域捷径与AI文本检测器的单向胜任域》

## 一句话贡献

用2×2×5×5的受控实验（register × prompt-contract × model × seed, n=2000 + 两轮先导 n=400）证明：**输出合约系统性地把LLM文本移入人类流形——机制随语域变化（正式语域=熵坍塌，随性语域=表层特征人化）；检测器的可靠胜任域收缩为单一格子（正式×粗糙模型）；"能力门槛"是语域条件机制而非通用阶梯**——并给出统一解释框架（规范松弛理论）与双向语域捷径的实证。

## 与相关工作的差异（定位节素材）

- FAILOpt/HC3/ArguGPT/GRACE：提示/域对检测的影响已知，但无"同内容跨语域×合约"的2×2因子设计，无能力阶梯复现，无胜任域量化
- Perplexity Trap (arXiv:2607.13044)：正式语域坍塌的理论——我们给出casual方向的镜像证据（捷径双向）+ 实验因果分离（合约 vs 语域 vs 交互）
- 1D Collapse/DivScore：编码器语域捷径已知——我们给出捷径的双向行为学证据与"胜任域"的操作化
- Sadasivan界：分布重叠上界——胜任域单格化是其在真实检测器上的具体化

## 章节结构（每节：论点 → 证据锚点 → 篇幅）

**1. 引言**（1页）
- 论点：AI检测讨论聚焦准确率，我们问"检测器在哪里还有效"；答案：一格
- 锚点：W4c漏检率表（图1：5×4热图，唯一浅色格）
- 贡献列表（4条）：2×2因子证据 / 规范松弛框架 / 双向语域捷径 / 开放探针集+回归体系

**2. 规范松弛理论**（1页）
- 论点：合约=把模型熵压向目标文体的人类流形；松弛位置由语域×模型能力决定
- 形式化：c(S)↑ ⇒ TV(M_c, H_reg)↓；预测：合约效应在有松弛处最大（formal×above-floor=熵松弛；casual×below-floor=表层松弛）
- 锚点：W4c交互对比表（formal A-B 减 casual C-D：above-floor +0.20~+0.39正交互；7B负交互-0.24——预测精确命中）

**3. 实验设计**（1.5页）
- 2×2×5模型×5种子；casual合约规范（emoji/标签/口语套语）；先导W4/W4b（n=320+80）
- 预注册三假设+分析口径（主题块bootstrap/Wilcoxon）；诚实记录：分析脚本三bug与修复（附录）
- **方法论补强清单落实**：matched-era切片（20篇2024-2026）对82% pre-2023主库；Wilson区间全量；固定操作点漏检率+AUROC双报告

**4. 结果**（2页）
- 4.1 主表+热图（唯一胜任格：GLM 9%/7B 15%）
- 4.2 H1-H3裁决（H1全复现含7B小幅；H2 ceiling-bound但7B/GLM显著；H3非单调→家族>参数量）
- 4.3 交互分析（规范松弛预测的证实）
- 4.4 双向语域捷径（casual free 60-100%漏检——新盲区的发现）

**5. 案例研究：一个可辩护检测器的生命周期**（1页，Opportunity资产变现）
- FN-1（单文档失败→根因三层→fixture回归）+ W7门控FAIL（对抗训练负结果的自动废弃）+ ADR-0001（理性不做的记录）
- 论点：在坍塌语域，检测器的正确产品形态是"仪器+边界声明"，不是判官

**6. 讨论与局限**（1页）
- 非独立性披露节：harness研究与本实验同作者同模型池——"双域检验"而非"独立验证"
- 局限：检测器单系统（未测商用检测器——Turnitin等无法受控评测）；阈值固定；GLM casual-D合约不合规混杂；7B种子可控/API侧服务端随机
- 治理含义：合约完备性与可检测性对偶→provenance路线的实证支撑（TraceLog推论）

**7. 结论**（0.25页）

## 图表清单

1. 图1：5模型×4格漏检率热图（主视觉）
2. 图2：规范松弛交互对比（formal-Δ vs casual-Δ散点，模型着色）
3. 图3：能力阶梯剂量反应（W4三点单调 vs W4c五点非单调——家族效应）
4. 表1：主表（均值[CI]/漏检率）
5. 表2：与相关工作对照
6. 附录：探针集构成/回归体系/分析bug记录/复现指令

## 写作执行契约

- fat-slim-writer：骨架（本件）→ fat稿（全证据倾倒）→ slim稿
- 每个数字可溯源到reports/或dataset/*/analysis.json；引用全部来自已验证的四路文献调研
- 写作前完成：方法论补强清单核对（matched-era✓/AUROC补算/生成器5✓）；era分层评测（主库vs matched-era切片的FPR对比）作为4.4小节素材
- 证伪器对齐：W4c零效应→本骨架作废（已发生的是"效应比假设更丰富"——骨架按实际数据更新，非作废）
