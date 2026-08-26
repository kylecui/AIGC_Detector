# 可辩护性计分卡（2026-08-18）

> **PRELIMINARY — 小样本预览**。人类侧 n=10（试批）、AI侧 n=240（实验语料）。
> 完整 W5 探针集（人类侧 60-80 篇）落地后重算；n<100 不报告 FPR@1%。

## D1 校准诚实（头号指标：高置信错误率）

| 侧 | n | 错误率 [Wilson 95%] | 高置信错误 (conf>0.8且错) | ECE(10桶) |
|---|---|---|---|---|
| AI侧（漏判） | 320 | 54.7% [49.21%, 60.05%] | 130 (40.6%) | 0.425 |
| 人类侧（误判,试批） | 62 | 19% [11.43%, 30.85%] | 8 | 0.164 |

高置信误判明细（人类侧）: 06-bank-ccb-system-maintenance, 06-tongbao-csrc-hlj-investor-2018, 07-chengqing-jianghuaiqiche-2015, 08-chengnuoshu-cac-tencent-hegui-2021, 08-yanzheng-shengming-amac, 10-chengqing-shengming-thu, 11-zhaohui-gac-honda-fit-2022, 12-zhaohui-subaru-2022

AI侧分模型漏判: Qwen2.5-7B-Instruct 18/80; GLM-4-9B-0414 51/120; DeepSeek-V3.2 106/120

## D2 盲区显性化（caveat覆盖，试批）

正式语域命中: 52/62 (84%) — 未命中为无公文套语的简短服务公告（已知缺口，W5全量校准轮处理）

## D5 FN-1 known-bad 回放

文档级判定 Human-written (0.89)；最高分段 p_ai = 0.856，段级证据浮出: ✅ PASS

## 结论（预览口径）

- AI侧漏判集中且方向一致（合约臂/强模型），与W4实验吻合——高置信错误集中在强模型合约文本。
- 人类侧试批出现高置信误判（建行公告 0.987）——D1 的直接反面教材，W3b权重切换须以此为约束。
- caveat覆盖：公文套语文本100%命中；简短服务公告缺口已登记。
- 发布门槛（正式版）：ECE不恶化 + caveat覆盖100%（正式语域全集）+ FN-1回放PASS。
