# Stage Contract (v0.3)

**对象**: 任何想为检测框架编写扩展stage的第三方作者。
**契约源码**: `src/aigc_detector/stages/contract.py`（`StageProtocol`，`@runtime_checkable`）
**参考实现**: `examples/stages/ttr_stage.py`（第三方视角示例：不import框架任何代码即可满足契约）

## 契约

```python
class StageProtocol(Protocol):
    stage_id: str
    def load(self) -> None: ...          # 幂等资源准备
    def unload(self) -> None: ...        # 释放；未加载时调用安全
    @property
    def is_loaded(self) -> bool: ...
    def predict(self, text: str, language: str | None = None) -> dict: ...
```

`predict` 返回dict，最少含：
- `p_ai: float ∈ [0,1]` — 该stage认为文本为AI生成的概率
- `label: str` — "AI-generated" | "Human-written"（stage自身阈值）
- `confidence: float` — 对自身label的信心
- 推荐：`evidence: dict`（原始特征/指标——可辩护检测的证据面）、`model: str`

**硬性纪律**：
1. `predict` 对任何输入不得抛异常——失败时返回中性结果（`p_ai=0.5, confidence=0.0` + evidence说明）。坏stage只降级证据，永不破坏判定。
2. 短文本/域外输入：诚实返回中性（见TTR示例的 `<40 tokens` 分支），不硬给分数。

## 两种集成角色

| 角色 | 进入方式 | 影响 |
|---|---|---|
| **Ensemble stage** | 需要pipeline权重工作（内置四阶段专属路线，见Roadmap） | 参与投票 |
| **Diagnostic stage** | Plan声明即插入（本契约的第三方扩展点） | **不投票**：结果以`breakdown["diagnostic_<id>"]`追加为可审计证据 |

## Diagnostic stage接入（三步，零核心改动）

1. 写stage类满足契约（放任何可导入位置；repo内建议`examples/stages/`）
2. `plans/default.yaml`追加：
   ```yaml
   diagnostic_stages:
     - id: my_stage
       impl: my_package.my_module:MyStage
   ```
3. 完成——PlanRunner实例化并在每次detect后把结果追加进breakdown（`_DiagnosticPipelineWrapper`组合，不改DetectionPipeline任何代码）

## 验收义务

接入diagnostic stage后必须跑：
- `uv run pytest tests/`（367+基线）
- `uv run python scripts/run_probes.py`（plan探针）
- 人工抽查一条detect响应确认`diagnostic_<id>`在场且evidence可读

## Roadmap（ensemble扩展，未开放）

新增投票stage需要：pipeline的stage_results注册、语言路由权重声明、探针集基线重校准（Wilson口径）、367测试回归——在无第三方需求前不开放此路径，避免表面可插拔实则校准失效。

## 为什么diagnostic不投票（设计立场）

可辩护检测的立场：证据可以廉价扩展（每个新信号都让`breakdown`更可审），但**判定权变更必须有校准数据背书**（W4-W4EN/W15的教训：权重与阈值的每次变更都经过探针集+门控）。契约把这两个面分开：evidence开放、verdict保守。
