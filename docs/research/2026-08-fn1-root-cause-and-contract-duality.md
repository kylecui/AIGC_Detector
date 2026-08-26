# FN-1 深度研究：正式语域失效根因、检测器与合约的对偶结构

**日期**: 2026-08-17
**状态**: 完成（根因分析+文献映射+交叉验证+实验设计）
**关联**: `DETECTOR_NOTES_2026-08.md` FN-1（数据记录与P4计划）；`kylecui/contract-driven-harness-study`（外部交叉研究）
**性质**: 研究文档，非操作手册。P4执行计划见 `.sisyphus/plans/fn1-countermeasures-and-paired-experiment.md`

---

## 摘要

2026-08-17 记录的假阴性 FN-1（AI起草的中文合规声明被判为 Human-written，置信度0.89）不是调参失误，而是三层机制的叠加：**流形坍塌**（结构性，信息论层面不可绕过）、**编码器形式性捷径**（训练性，可修复）、**聚合稀释**（工程性，最廉价可修复）。外部文献为每一层提供了独立验证。

更重要的发现是概念性的：本次失效与合约驱动生成研究（contract-driven-harness）构成**对偶**。统计检测器利用的是生成过程的残余熵（未合约化的模型松弛）；显式合约通过外部化义务消灭同一熵。因此**合约确定性与来源不可检测性是同一现象的两把仪表**：当合约完备时，输出分布收敛于目标文体的人类分布（TV距离→0），零样本统计检测趋于随机，来源证据必须从文本迁移到轨迹（TraceLog/process evidence）。

本文档记录根因、证据链、对偶论证，并给出配对实验设计（自由生成 vs 合约约束生成的可检测性差异测量）。

---

## 1. 案例数据（引用自 FN-1）

送检文本：`docs/software-copyright/软件合法合规及原创性声明.md`（2029字符，剥离使用说明注释块后），AI全程起草、轻度人工编辑。公文体：编号条款、法定模板措辞（"特此声明"等）、信息表格、落款签章区。

| 阶段 | p_ai | 判定 |
|---|---|---|
| statistical (Wenzhong-110M) | 0.0108 | human ❌ |
| linguistic (14特征XGB) | 0.2157 | human ❌ |
| encoder (chinese-roberta+LoRA v2) | 0.0297 | human ❌ |
| binoculars (Qwen2-7B, 4-bit) | 0.3431 | human ❌ |
| **ensemble** `{stat:.10, ling:.10, enc:.60, bino:.20}` | **0.1091** | **Human-written 0.891 ❌** |

ZH阈值0.47。四个轴独立失效——非权重伪影。分段视图：8段中7段p_ai 0.01–0.21，**第8段（排比收尾条款）0.8560**，被加权均值稀释。

语言诊断：`meso=10.0（满分）`——表格+条款的异质段落形态触发段落长度方差加分，meso层将其读作"人类特征"。

---

## 2. 三层根因与证据映射

### 2.1 第一层：流形坍塌（Perplexity Trap）——结构性主因

**机制**：Banerjee 2026（arXiv:2607.13044）形式化三个条件：
- **C1** 输出受外部语法约束；
- **C2** 约束同时作用于人类与AI输出；
- **C3** 约束在至少一个语言轴（词汇/句法/修辞）上压缩变异。

三条件齐备 ⇒ 人类与AI的困惑度分布坍塌到同一低熵流形 ⇒ 似然类检测失去可分性。该论文证明此失效是**检测器内部的**：曲率修正、对比打分、领域适配头、似然比重校准均无法恢复可分性。

**实证镜像**：EPO专利权利要求（法定"清楚简洁"要求=C1-C3）上，零样本检测器claim级FPR全部>60%：Binoculars 78.3%、Fast-DetectGPT 61.3%、DetectGPT 80.5%。跨IPC复现（A61K/C07D/F03D）平均FPR 84.6%；换生成模型（Qwen2.5-3B）、换打分模型（Falcon-7B/GPT-J-6B on H100）均不缓解。

**与FN-1的关系**：我们的声明满足C1（公文体法定格式）、C2（CPCC补正要求同等约束人类律师与AI）、C3（模板措辞+条款结构压缩修辞空间）。EPO研究测得的是**人类正式文本被误判为AI**（FPR侧），我们测得的是**AI正式文本被误判为人类**（FNR侧）——同一坍塌在阈值两侧的相反症状。这一镜像本身就是"结构性而非统计性失效"的证据：如果是训练偏差，两侧症状不会在跨语言、跨检测器族的情况下对称出现。

**信息论定界**：Sadasivan et al.（arXiv:2303.11156）：任意检测器 `AUROC ≤ 1/2 + TV(M,H) − TV(M,H)²/2`。强约束语域按构造收窄TV。Chakraborty et al.（arXiv:2304.04736）：多样本池化（同作者/同语料）经Chernoff信息恢复可分性，n = Ω(1/δ²)。含义：坍塌语域中单文档判决信息论受限，语料级判决保留空间。

**推论**：在C1-C3齐备的语域，零样本统计检测的终局是来源证明（水印/轨迹/过程日志），不是更好的似然统计。

### 2.2 第二层：编码器形式性捷径——训练性，可修复

**机制**：Smirnov 2026（"The 1D Collapse in AI Text Detection"，Zenodo 19399532）：测试的8+检测器（RoBERTa/ELECTRA/DeBERTa/DistilRoBERTa/ALBERT，5个独立实验室）全部将768维表征坍缩至约一维，且该维度与**文本形式性**对齐（cos 0.73–0.99）而非作者身份。坍缩是预训练内禀的、检测任务特有的、训练中稳定的，集中在transformer层8-12。类间轴与形式性轴cos>0.6即发生坍缩。

**代码侧确认**（本次盘点）：encoder-zh训练数据（HC3）人类侧为知乎/百科式正式问答（crawler.py:172-182，SOURCE_DOMAIN_MAP将law/baike均映射general），AI侧为2023年ChatGPT闲聊；Route B过采样的是教科书语域。**两侧均无公文语域**。"正式语域→人类"在训练分布内是有效判别特征，在公文上正好反噬。

**已证有效的修复**（同论文）：梯度反转对抗形式性训练（形式性对齐cos 0.98→0.45，EvoBench TPR +5.6pp）；AI对比训练（跨任务域聚合AI表征）。

**反面警告**（DivScore, ACL 2025）：朴素领域适配可能**恶化**重叠——适配后打分头内化了目标域AI的表面统计，人机分布反而更近（overlap 0.585→0.780，|d| 0.96→0.49）。任何公文语域重训练必须**域内人机双侧平衡**。

### 2.3 第三层：聚合稀释——工程性，最廉价可修复

**机制**：arXiv:2605.06294（Log-Likelihood, Simpson's Paradox）：token/段级似然的朴素均值聚合在统计结构异质的区域间造成Simpson悖论，强局部信号被摧毁。FN-1是教科书实例：0.856的排比条款段（全文约束最松、模型自由度最高处）被7个低分段稀释至0.109。

**先行工作**：
- WaterSeeker（arXiv:2409.05112）："先定位后检测"，粗到细，保留文档均值会丢失的局部信号；
- GigaCheck（ACL 2026）：DETR式span回归定位AI区间，文档分类与span定位解耦，跨域泛化强；
- FairOPT（arXiv:2502.04528）：分组自适应阈值（按形式性/长度分组）降低跨组差异27.4%，总体精度损失<0.1%；
- RAID（ACL 2024）：域内校准先行，固定FPR报告，禁止跨域朴素0.5阈值。

**修复路径**（详见计划文档W1/W2/W3）：修复`segment_index`序列化bug（routes.py:93，当前返回seg#None，阻塞一切段级特性）；"局部AI痕迹"次级输出（max/top-k段p_ai+计数，作为辅助信号而非主判定，规避长人类文档单段高分的FP风险）；公文体register门控（语域检测→binoculars主导权重+置信区间加宽+UI提示，**明确禁止朴素降阈值**——那会把FPR侧镜像症状引进来）。

---

## 3. 检测器与合约的对偶结构（交叉验证）

### 3.1 合约驱动研究要点（kylecui/contract-driven-harness-study）

论文《Contracts as Task Skeleton: Externalizing Implicit Obligations for Bounded LLM-Agent Determinism》核心论断：bounded任务上，**合约完备性是agent确定性的binding constraint，而非模型智能**。证据：冻结G9协议下Qwen3-8B 40/40严格通过（Wilson [0.912,1.000]）；同一模型0/40→40/40经四次repair-loop迭代，零重训零提示调参；GLM-4-9B 30/40暴露可修复的paraphrase脆弱点；Qwen2.5-7B 0/40结构性地板。合约栈七对象：TaskSpec/MemorySlice/EvidenceBundle/OutputContract/WorkflowGate/**TraceLog**/ValidatorGate。失败三分类：合约缺陷/评估器缺陷/模型失效。范围明确限定单任务族（受控状态变异），否认普适性。

### 3.2 对偶论证

两个项目测量同一个量——**给定任务规约后输出的残余熵**（未合约化的模型松弛）：

- **检测器利用残余熵**：困惑度、爆发度、Binoculars交叉困惑度比值、文体方差特征——全部吃自由生成留下的统计松弛。AI文本检测文献的攻击面分析（paraphrase攻击Krishna et al. NeurIPS 2023，DetectGPT 70.3%→4.6%；风格对齐DPO/MASH攻击成功率92%；提示约束致F1方差σ=14.4，HowYouPromptMatters EMNLP 2024）本质上都是**攻击者主动压缩自己输出的熵以拟合人类文体流形**。
- **合约消灭残余熵**：外部化义务=把松弛收归系统层。合约越完备，输出越由规约决定而非模型采样决定。

**命题（对偶）**：`合约确定性与来源统计不可检测性是同一现象的两把仪表`。形式化：设任务规约S的完备度为c(S)，模型自由生成分布M₀，合约约束后分布M_c，目标文体人类分布H。c↑ ⇒ TV(M_c, H)↓（合约把模型推向文体规约本身，而人类好文本本就合规约）⇒ 检测AUROC上界↓。当c达到"完备"（输出合约覆盖结构/措辞/格式/边界），M_c在文本统计层面与H可交换。

**FN-1即此命题的实测**：声明是在CPCC补正要求这个"输出合约"（结构、必备条款、法定措辞、表格schema、落款）约束下生成的。合约完备度足够高，以至于工业级四阶段检测器测不出残余松弛。这是contract-driven论文"binding constraint"论断迄今最强的旁证——**不是行为层面（40/40），而是分布层面（与人类文体统计不可区分）**。

### 3.3 用论文自身的失败分类法反审FN-1

以合约论文§3.4三分法评估"检测器把AI文判为人类"这一violation：义务（检测法律语域AI文本）是否在evaluator的declared scope（训练数据覆盖）内？**否** ⇒ 按分类这是**评估器缺陷（覆盖缺陷）**，非检测器族失效。修复动作恰好是repair loop：命名缺失义务→FN-1文档做成known-bad fixture→扩展覆盖→重验golden/known-bad锚点。本项目的P4流程不知不觉运行在合约论文的协议里。

### 3.4 TraceLog推论（来源证明的迁移）

合约栈内化provenance（TraceLog：决策轨迹/证据绑定/阶段门）正因为统计provence（可检测性）被合约外部化消灭。推论：

> **当合约完备时，来源证据从文本统计迁移到执行轨迹。** 统计检测未合约化的松弛；合约文档化已合约化的意图。二者互补覆盖全域：无合约处用检测器测松弛，有合约处用TraceLog验意图。对AIGC治理的含义：高合规场景（法律/专利/公文，恰是C1-C3语域）的"AI含量"审计不应依赖文本检测器，应依赖生成过程留痕（API日志/编辑历史/水印），文本检测器退居无留痕场景。

---

## 4. 哲学裁定：失败还是成功？

**双重认定，但不对称**。

### 4.1 失败面（不可豁免）

检测系统的操作目的是判别来源。它以0.89置信度输出了错误答案。若这是审核/查重场景，文档已通过。以"其实是成功"豁免 = 动机性推理。FN必须记录为FN（已做），修复义务照常成立（计划已排）。**这一面的坚持是检测器项目的信誉基础**。

### 4.2 成功面（真实且更深）

 FN-1为合约研究提供了行为指标之外的**分布指标**：检测器意外充当了"未合约化残余熵"测量仪，读数接近零。两个独立项目（一个追求可靠合约生成，一个追求检测AI文本）在同一文档上相遇，后者证明前者达到了自己都不知道的完备度。这种跨项目证伪/证实是研究里最稀有的事件之一。

### 4.3 代价与取舍（必须正视的推论）

同一机制让弱模型可靠，也让**任何**模型的输出与文体合规的人类文本统计可交换。这不是合约研究的缺陷，而是其成功的定义性后果——就像加密的完善保密性以输出均匀为代价。接受此代价的理由：合约路线的可靠性不依赖统计不可区分性，而依赖TraceLog的可审计性。**可审计性 > 可检测性**是更工程化的来源保证。

### 4.4 认识论脚注

"检测失败作为合约成功"的读法依赖一个外部条件：文本确由AI在合约约束下生成（本例为真——我们是生成者）。对第三方审计者，无法先验知道一份低p_ai公文是"人类写的"还是"完备合约下的AI输出"——这正是C1-C3语域中检测器失效的表述形式。因此该"成功"是不可传播的私人知识，除非配TraceLog披露。**合约研究的对偶胜利无法被外部验证，除非主动交出轨迹。**这把4.3的推论推到极点：合约完备性的证明责任最终落在披露制度上，而非文本上。

---

## 5. 配对实验设计（已执行完毕，结果见DETECTOR_NOTES_2026-08.md W4节）

**结果概要**（n=40主题×3模型×2臂=240样本，2026-08-17）：
- **能力门槛证实**：合约逃逸仅出现在above-floor模型上——GLM-4-9B Δensemble=+0.487（sign p=2e-08, d_z=1.31, 37+/3−）；DeepSeek-V3.2 Δ=+0.230（p=7e-04）；below-floor的Qwen2.5-7B零效应（20+/19−）
- **最干净的结果**：同一合约下，B臂p_ai随合约执行能力单调下降：**0.74（7B）→ 0.34（GLM）→ 0.10（DeepSeek）**
- 阶段分解：statistical坍缩最狠（GLM d_z=2.44，困惑度压缩是核心机制）；linguistic在公文语域结构性失明（全部n.s.——与FN-1 meso满分互相印证）；binoculars响应最弱
- 对偶论证获得实证：可检测性距离（A−B）作为"未合约化残余熵"的测量轴，随合约执行能力单调——给contract-driven研究提供了行为指标之外的分布指标

**动机**：文献裁决指出"约束vs自由生成的受控配对证据"稀缺（HowYouPromptMatters测了提示约束的方差效应，但无"同语义内容、约束vs无约束"的配对设计直接测检测响应）。该实验同时服务两个项目。

**设计**：
- **主题池**：中文公文主题（声明/承诺书/情况说明/合规承诺等），N=40–100；
- **Arm A（自由提示）**：`请就{主题}写一份{文种}`，最小约束；
- **Arm B（合约约束）**：完整spec——必备条款顺序、法定措辞（"特此声明"等）、表格schema、落款区、字数带、格式约束，模拟OutputContract；
- 同一模型、temperature固定（0.7），同主题成对生成；
- **检测**：本地四阶段API，记录全部分解；
- **分析**：配对Wilcoxon符号秩检验 + 配对t检验（A vs B的ensemble p_ai与各阶段p_ai），效应量（Cliff's delta / Cohen's d for paired）；
- **假设**：H1: B臂ensemble p_ai系统性低于A臂；H2: B臂段间方差更小（熵压缩的空间证据）；探索性：四阶段中哪个阶段对约束最敏感（预期statistical最敏感——它直接测熵）。
- **副产品**：配对语料直接成为P4-Tier2 `legal_declaration_zh`探针集的AI侧首批样本。

**Pilot**：10主题验证管线后再决定全量。

---

## 6. 参考文献（按本文档引用顺序）

1. Banerjee 2026. The Perplexity Trap: When Patent Law Makes Human Writing Look Like AI. arXiv:2607.13044
2. Sadasivan et al. 2023. Can AI-Generated Text be Reliably Detected? arXiv:2303.11156
3. Chakraborty et al. 2023. On the Possibilities of AI-Generated Text Detection. arXiv:2304.04736 (ICML 2024)
4. Smirnov 2026. The 1D Collapse in AI Text Detection. Zenodo 19399532
5. DivScore 2025. Zero-Shot Detection of LLM-Generated Text in Specialized Domains. ACL 2025
6. Log-Likelihood, Simpson's Paradox, and the Detection of Machine-Generated Text. arXiv:2605.06294
7. WaterSeeker 2024. arXiv:2409.05112
8. GigaCheck 2026. Detecting LLM-Generated Content via Object-Centric Span Localization. ACL 2026
9. FairOPT 2025. Group-Adaptive Threshold Optimization. arXiv:2502.04528
10. RAID 2024. A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. ACL 2024
11. Krishna et al. 2023. Paraphrasing evades detectors (DIPPER). NeurIPS 2023, arXiv:2303.13408
12. How You Prompt Matters. EMNLP 2024 Findings. aclanthology.org/2024.findings-emnlp.156
13. MASH 2026. ACL 2026 Findings. aclanthology.org/2026.findings-acl.1487
14. Liang et al. 2023. GPT detectors biased against non-native English writers. Patterns
15. Hans et al. 2024. Binoculars. ICML 2024
16. Kirchenbauer et al. 2024. On the Reliability of Watermarks. ICLR 2024, arXiv:2306.04634
17. kylecui/contract-driven-harness-study. Contracts as Task Skeleton (v5 draft). 2026
18. DOMINO 2024. Guiding LLMs The Right Way. ICML 2024, arXiv:2403.06988
