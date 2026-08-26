# 运行规格与SLA声明

**版本**: 0.1（2026-08-21）
**定位**: 本文档声明 AIGC_Detector v0.1 的**运行规格与运营边界**。产品级 = 交付到声明的规格，而非交付到想象：以下所有数字为本项目单机实测口径并附证据来源；不在本文档中的能力，就是 v0.1 不具备的能力。**检测质量边界（哪些语域可靠、哪些是盲区、误判率多少）由 `docs/capability-statement.md` 声明，本文档不重复——能力文档管"判得准不准"，本文档管"跑得动、怎么跑"。**

---

## 一、定位声明：这是什么

AIGC_Detector v0.1 是**单租户、自托管的中英文AI生成文本检测仪器**：

- 部署形态：本地GPU机器上由 uvicorn 启动的 FastAPI 进程，操作者为本机/局域网内的单一用户；
- 使用形态：交互式单文档检测（WebUI 或 REST API），结果附带分段证据与置信度；
- 责任主体：操作者即管理员——备份、重启、模型与校准工件更新均由部署者自行承担；
- 设计目标：研究与审核辅助**仪器**（instrument），不是在线**服务**（service）。

**它不是**：

| 不是什么 | 说明 |
|---|---|
| 多租户SaaS | 无租户隔离、无配额管理、无组织概念；并发模型按单用户设计 |
| 有SLA背书的服务 | 无可用性承诺、无监控告警、无补偿概念（见第五节） |
| 批量处理管道 | 无批量API、无任务队列、无报告导出；是否做属产品决策（release-plan-v2 #14） |
| 带访问控制的服务 | v0.1 无认证/授权。auth 与公网暴露部署指引一同在 v0.2 提供（release-plan-v2 P1 #10） |

## 二、性能规格（声明值 = 单机实测口径）

| 指标 | 声明值 | 证据来源 |
|---|---|---|
| 稳态吞吐 | **GPU天花板约14.7 req/min**（1:1:1混合负载实测，90s窗口22请求）；**服务层单IP上限10 req/min**（限流）。有效吞吐=min(两者)，非承诺值 | `reports/perf_baseline.json`（2026-08-26实测，`scripts/perf_baseline.py`可复现）；限流见 `src/aigc_detector/api/routes.py` |
| 单请求延迟·中文非正式（热机） | 约0.3-1.3s | 同上盘点；casual链路跳过floor的单例254ms见 `reports/w15_gate_review.md` §2 |
| 单请求延迟·英文（热机） | 约1s | 同上盘点 |
| 单请求延迟·中文正式文书（命中W15下限） | **额外+17-23s**（实测17.6-23.1s；Qwen2-7B对，4-bit，12GB共享GPU） | `reports/w15_gate_review.md` §2 延迟实测 |
| 冷启动（进程内首次检测） | zh路径约70s（全模型链加载）；en路径约13s；后续binoculars补跑约3s | `reports/perf_baseline.json`（单次进程实测；冷启动方差大，视为量级非精确值） |
| 显存 | 12GB设计包络；全程实测峰值11.4GB | `src/aigc_detector/models/manager.py`（4-bit量化+LRU逐出的设计目标）；release-plan-v2仲裁记录 |

**读数须知**：

- 延迟数字为热机单请求口径；冷启动后的首个请求、语言切换后的首个请求显著更慢（见冷启动行）。
- 正式文书的+17-23s是**用检测力换的**（W15下限规则强制补跑binoculars），不是性能缺陷；规则取舍见 `reports/w15_gate_review.md`。
- 吞吐瓶颈在GPU侧（信号量=2与binoculars耗时），API限流（10/min/IP，`src/aigc_detector/api/routes.py` `@limiter.limit("10/minute")`）高于实测吞吐——放宽限流不提高吞吐。
- 性能数字已有回归脚本（`scripts/perf_baseline.py` → `reports/perf_baseline.json`），重跑即刷新本表。原"约3 req/min"为会话估算，实测GPU天花板更高（14.7 rpm混合），已按实测修正。

## 三、运行环境规格

| 项目 | 规格 | 依据 |
|---|---|---|
| GPU | 12GB VRAM设计包络（RTX 3060级别）；4-bit量化+LRU逐出 | `README.md` Model Bootstrap；`src/aigc_detector/models/manager.py` |
| 同机占用 | 桌面应用显存占用需计入：实测会话约3-4.4GB | `DETECTOR_NOTES_2026-08.md`；release-plan-v2（峰值11.4GB含此背景） |
| 磁盘 | 基础模型约4-8GB（`scripts/download_models.py`）；可选binoculars模型对约28GB（`scripts/prefetch_binoculars.py`，约14GB/语言，后台断点续传） | `README.md` |
| 网络 | 首次运行需HuggingFace可达；生产建议离线 `models/` 树 | `README.md`；`DEPLOYMENT.md` §9.2 |
| 验证环境 | Windows + RTX 3060 12GB + Python 3.12/uv | `DEPLOYMENT.md` §4 |
| 容器化 | 最小Dockerfile为P0条目（CPU路径验证）；GPU容器验证排v0.2窗口 | release-plan-v2 #8 / P1 #12 |

## 四、升级触发条件（本文档何时必须重写）

以下任一事件发生时，本文档的定位假设或数字失效，必须更新：

| 触发事件 | 失效假设 | 对应计划条目 |
|---|---|---|
| 第一个真实外部用户 | "单租户、单操作者"假设失效；吞吐与支持模型需重估 | release-plan-v2 P1 #10-#12（auth / metrics / Docker GPU验证） |
| 任何公网暴露部署 | "无认证、局域网信任"假设失效；auth、CORS、上传加固、日志留存全部前置 | release-plan-v2 P1 #10（公网暴露前置批次，v0.2邀请测试前） |
| 批量API需求出现 | 交互式单文档定位失效；GPU天花板14.7 req/min与批量吞吐的差距必须显式决策 | release-plan-v2 P2 #14（产品决策项：进路线图或明确不做） |
| 性能数字重测 | 第二节全部声明值以新基线为准 | DEVPLAN 4.13 性能测试任务 |

## 五、明确的不承诺

- **不承诺可用性**：本地进程，无监控、无告警、无自动恢复；进程挂了由操作者重启。
- **不承诺并发**：GPU并发信号量=2（`src/aigc_detector/api/routes.py`）；第3个请求排队，120秒超时返回503。
- **不承诺多租户安全**：无认证、无授权、无租户隔离；局域网即是信任边界。
- **隐私处理仅一条**：请求文本不写入日志（脱敏门禁强制，release-plan-v2 P0 #7）。除此之外不承诺数据保留策略、销毁流程或合规认证。
- **不承诺吞吐与延迟的SLA值**：第二节是仪器规格（instrument spec），不是服务合同（service contract）；数字用于容量规划，不用于赔付计算。
- **不承诺跨环境复现**：所有数字来自单台RTX 3060 12GB实测；不同GPU、驱动或桌面并发负载下数字会变。
- **不承诺更新节奏**：模型、阈值与校准工件随研究进展不定期更新，无定期发布或安全补丁SLA。

## 六、版本与分工

- 本文档版本 0.1，日期 2026-08-21；数字随重测与升级触发条件更新（见第四节）。
- **分工**：检测能力边界（语域可靠性、盲区、误判率、审核决策流程）→ `docs/capability-statement.md`；运行边界（吞吐、延迟、环境、支持方式）→ 本文档。引用检测结论前先读能力文档；规划部署前先读本文档。
- 相关材料：`docs/capability-statement.md`、`reports/w15_gate_review.md`、`DEPLOYMENT.md`、`.sisyphus/plans/release-plan-v2.md`、`DETECTOR_NOTES_2026-08.md`。
