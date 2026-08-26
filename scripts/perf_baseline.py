"""D1/D2: scripted performance baseline — cold start + latency/throughput.

Fills the two [待测] markers in docs/sla-statement.md:
  D1 cold start: per-language lazy-load wall time (models uncached in RAM)
  D2 latency: p50/p95 per scenario (zh-casual, en-casual, zh-formal w/ floor)
     + sustained throughput over 60s mixed load
Run with a RESTARTED process for honest cold start (fresh model loading);
latency/throughput phases run warm. GPU: 12GB shared (desktop ~3.3GB).

Usage: uv run python scripts/perf_baseline.py [--skip-cold]
Output: reports/perf_baseline.json (+ console)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

ZH_CASUAL = (
    "这家店真的绝了，排队两小时但味道完全值得。豚骨拉面汤底浓郁到离谱，"
    "饺子也是满分水平，服务员看我们等太久还送了茶，好感度直接拉满。"
    "下次带朋友一起来，强烈推荐工作日下午去，人稍微少一点。总之就是好吃！"
) * 2

EN_CASUAL = (
    "ngl this place is absurd. waited 2 hours but tbh the tonkotsu literally "
    "slapped. lowkey coming back Tuesday. also the gyoza was fire, ten out of "
    "ten, no notes. the vibes were immaculate and the staff were so nice."
) * 2

ZH_FORMAL = (
    "上海景治智能科技有限公司（以下简称“本公司”）郑重声明：依据《生成式人工智能"
    "服务管理暂行办法》有关规定，现就本公司软件作品的合法合规性声明如下。"
    "一、软件基本情况：本公司独立开发完成中英文AI生成文本检测系统V1.0，"
    "二、原创性承诺：该软件源代码均为本公司开发人员独立完成。三、合规使用说明。"
    "特此声明。本公司承诺严格遵守相关法律法规，如有违反愿承担相应责任。"
    "上海景治智能科技有限公司　＿＿＿＿年＿＿月＿＿日"
)


def pct(xs, p):
    xs = sorted(xs)
    i = min(len(xs) - 1, int(p * len(xs) + 0.5))
    return xs[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cold", action="store_true")
    args = ap.parse_args()

    report: dict = {"timestamp": datetime.now(UTC).isoformat()}

    t0 = time.time()
    from evaluate_paired_experiment import build_pipeline

    pipeline = build_pipeline()
    build_s = time.time() - t0
    report["pipeline_construct_s"] = round(build_s, 2)

    # ---- D1 cold start: first detection per path loads models ----
    if not args.skip_cold:
        for name, text in [("zh_casual", ZH_CASUAL), ("en_casual", EN_CASUAL)]:
            t = time.time()
            pipeline.detect(text)
            report[f"cold_start_{name}_s"] = round(time.time() - t, 2)
        # zh formal: includes W15 forced binoculars on first hit
        t = time.time()
        pipeline.detect(ZH_FORMAL)
        report["cold_start_zh_formal_s"] = round(time.time() - t, 2)

    # ---- D2 warm latency percentiles (n=12 each) ----
    for name, text in [("zh_casual", ZH_CASUAL), ("en_casual", EN_CASUAL),
                       ("zh_formal_floor", ZH_FORMAL)]:
        xs = []
        for _ in range(12):
            t = time.time()
            pipeline.detect(text)
            xs.append((time.time() - t) * 1000)
        report[f"latency_{name}_ms"] = {
            "p50": round(pct(xs, 0.5)), "p95": round(pct(xs, 0.95)),
            "mean": round(statistics.mean(xs)),
        }

    # ---- D2 sustained throughput: 90s mixed (2 casual : 1 formal) ----
    n_ok = 0
    lat = []
    t_end = time.time() + 90
    while time.time() < t_end:
        for text in (ZH_CASUAL, EN_CASUAL, ZH_FORMAL):
            if time.time() >= t_end:
                break
            t = time.time()
            pipeline.detect(text)
            lat.append(time.time() - t)
            n_ok += 1
    dur = 90
    report["throughput_90s"] = {
        "requests": n_ok, "rpm": round(n_ok / dur * 60, 1),
        "mix": "zh-casual/en-casual/zh-formal 1:1:1 (no rate limiter — GPU bound)",
        "achieved_rpm_note": "service adds 10/min/IP limiter; this measures GPU ceiling",
    }

    out = Path("reports/perf_baseline.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
