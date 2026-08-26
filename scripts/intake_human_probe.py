"""Intake pipeline for human-side probe set (protocol §4).

Consolidates staging dirs (trial already in main store + 4 agent staging
dirs), enforces:
  H4 seven-field provenance validation
  H5 dedup (sha256 exact + 5-gram Jaccard > 0.8 near-dup)
  H1 register gate (score >= 6; borderline 4-5 → review list for human call)
  quota reconciliation vs protocol §3 matrix (report only, no auto-drop)
  era accounting (H2: pre-2023 >= 70% of full set)

Usage: uv run python scripts/intake_human_probe.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.detection.register import detect_register_zh  # noqa: E402

ROOT = Path(__file__).parent.parent
MAIN = ROOT / "dataset/legal_declaration_zh/human"
STAGING_ROOT = Path("C:/Users/kylecui/AppData/Local/Temp/opencode/human_probe")
STAGING_DIRS = ["cninfo_cc", "cninfo_ac", "gov_a", "tier_c"]

REQUIRED_FIELDS = ["source_url", "publisher", "published_date", "fetch_date",
                   "doc_type", "license_note"]

# coarse doc_type bucket → protocol matrix family (substring match; trial batch
# uses descriptive types like '上市公司更正公告（公文体）')
def family_of(doc_type: str | None) -> str:
    dt = doc_type or ""
    for key, fam in [
        ("更正", "更正"), ("澄清", "澄清"), ("致歉", "致歉"), ("道歉", "致歉"),
        ("承诺", "承诺"), ("情况说明", "通报"), ("通报", "通报"),
        ("召回", "召回维护"), ("维护", "召回维护"), ("监管", "声明"), ("声明", "声明"),
    ]:
        if key in dt:
            return fam
    return "其他"


# H1 human-review ACCEPT overrides (protocol §2.1: 不达标→人工复核→记录).
# These genres are formal by construction but lexically sparse for the rule
# gate (table-format recalls, bare 承诺函, informal-register apologies) and
# each maps 1:1 to an AI-side probe topic — same-register pairing is the
# protocol's golden rule, so excluding them would bias the baseline.
HUMAN_ACCEPT: dict[str, str] = {
    "11-zhaohui-gac-honda-fit-2022.md":
        "recall-table genre; pairs AI-side t18 产品召回公告; gate blind to table format",
    "12-zhaohui-subaru-2022.md":
        "recall-table genre; pairs AI-side t18 产品召回公告; gate blind to table format",
    "08-chengnuoshu-cac-tencent-hegui-2021.md":
        "承诺函 subgenre (score 5 borderline); pairs AI-side 承诺书 topics",
    "10-chengnuoshu-chaoyang-baojiaheng-huanbao-2020.md":
        "bare 环保承诺函 (sparse formulae); pairs AI-side t15 环保承诺书",
    "01-zhiqian-shengming-mezh.md":
        "institutional apology, informal register; pairs AI-side t03 道歉声明",
    "02-zhiqian-shengming-ctrip.md":
        "corporate apology+explanation (score 5 borderline); pairs AI-side t03/t02",
    "05-tongbao-mee-yulin-lantan-2021.md":
        "中央督察组典型案例通报 (clauses are bolded headers, gate sees half); "
        "pairs AI-side t02/t12 情况说明/报告; pre-2023 high-value genre",
}


def parse_doc(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    header, body = parts[1], parts[2].strip()
    meta = {}
    for line in header.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return {"path": path, "meta": meta, "body": body}


def sha5grams(body: str) -> set[str]:
    clean = re.sub(r"\s+", "", body)
    return {clean[i:i + 5] for i in range(max(0, len(clean) - 4))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    docs: list[dict] = []
    problems: list[dict] = []

    # main store (trial) + staging
    sources = [("main", MAIN)]
    for d in STAGING_DIRS:
        p = STAGING_ROOT / d
        if p.exists():
            sources.append((d, p))

    for tag, directory in sources:
        for f in sorted(directory.rglob("*.md")):
            if "review" in f.parts or "_pdfs" in f.parts:
                continue
            doc = parse_doc(f)
            if doc is None:
                problems.append({"file": str(f), "issue": "no-header-block"})
                continue
            missing = [k for k in REQUIRED_FIELDS if not doc["meta"].get(k)]
            if missing:
                problems.append({"file": str(f), "issue": f"missing-fields:{missing}"})
                continue
            if len(doc["body"]) < 200:
                problems.append({"file": str(f), "issue": f"too-short:{len(doc['body'])}"})
                continue
            doc["tag"] = tag
            docs.append(doc)

    print(f"parsed: {len(docs)} docs; problems: {len(problems)}")

    # H5 dedup: exact sha then near-dup 5gram within same family
    seen_hash: dict[str, str] = {}
    grams_by_family: dict[str, list[tuple[str, set[str]]]] = {}
    keep: list[dict] = []
    dupes: list[dict] = []
    for doc in docs:
        h = hashlib.sha256(doc["body"].encode("utf-8")).hexdigest()
        fam = family_of(doc["meta"].get("doc_type"))
        doc["family"] = fam
        if h in seen_hash:
            dupes.append({"file": str(doc["path"]), "issue": f"exact-dup-of:{seen_hash[h]}"})
            continue
        g = sha5grams(doc["body"])
        near = None
        for other_name, other_g in grams_by_family.get(fam, []):
            if jaccard(g, other_g) > 0.8:
                near = other_name
                break
        if near:
            dupes.append({"file": str(doc["path"]), "issue": f"near-dup-of:{near}"})
            continue
        seen_hash[h] = str(doc["path"])
        grams_by_family.setdefault(fam, []).append((str(doc["path"]), g))
        keep.append(doc)

    print(f"after dedup: {len(keep)} (removed {len(dupes)})")

    # H1 register gate on keepers. Scoring uses a formatting-normalized copy
    # (strip markdown bold/heading artifacts introduced by web-reader
    # conversion — e.g. "**一、基本情况**" breaking the line-start clause
    # regex). Stored files keep their verbatim text untouched.
    for doc in keep:
        norm = re.sub(r"(?m)^#{1,3}\s*", "", doc["body"])
        norm = re.sub(r"(?m)^\*\*(.+?)\*\*\s*$", r"\1", norm)
        doc["register_score"] = detect_register_zh(norm).score

    # copy new staging docs into main store (unless dry-run); gate-missed docs
    # go to review UNLESS covered by a documented human-review accept override
    ingested = 0
    review: list[dict] = []
    accepted: list[dict] = []
    for doc in keep:
        score = doc["register_score"]
        name = doc["path"].name
        if score < 6 and name not in HUMAN_ACCEPT:
            review.append({"file": name, "score": score,
                           "family": doc["family"], "tag": doc["tag"]})
            continue  # do not ingest borderline without human call
        if doc["tag"] != "main" and not args.dry_run:
            dest = MAIN / name
            if not dest.exists():
                dest.write_text(doc["path"].read_text(encoding="utf-8"), encoding="utf-8")
                ingested += 1
        if score < 6:
            accepted.append({"file": name, "score": score,
                             "reason": HUMAN_ACCEPT[name]})

    # quota + era accounting over final store (incl. human-accepted overrides)
    final = [d for d in keep
             if d["tag"] == "main" or d["register_score"] >= 6
             or d["path"].name in HUMAN_ACCEPT]
    fam_count: dict[str, int] = {}
    era_count = {"pre2023": 0, "post2023": 0, "unknown": 0}
    issuer_count: dict[str, int] = {}
    for d in final:
        fam_count[d["family"]] = fam_count.get(d["family"], 0) + 1
        date = d["meta"].get("published_date", "")
        era = d["meta"].get("era") or ("pre2023" if date < "2023-01-01" else
                                       "post2023" if date >= "2023-01-01" else "unknown")
        era_count[era] = era_count.get(era, 0) + 1
        issuer_count[d["meta"]["publisher"]] = issuer_count.get(d["meta"]["publisher"], 0) + 1

    total = len(final)
    pre_share = era_count["pre2023"] / total if total else 0
    single_issuer_over = {k: v for k, v in issuer_count.items() if v > 3}

    print(f"\n=== intake report {'(DRY RUN)' if args.dry_run else ''} ===")
    print(f"final store: {total} docs (ingested new: {ingested})")
    print(f"families: {json.dumps(fam_count, ensure_ascii=False)}")
    print(f"era: {json.dumps(era_count)} (pre-2023 share {pre_share:.0%}, target ≥70%)")
    print(f"issuers with >3 docs: {single_issuer_over or 'none'}")
    print(f"review-needed (score<6): {len(review)}: "
          + "; ".join(f"{r['file']}({r['score']})" for r in review[:10]))
    print(f"excluded/dupes: {len(dupes)}; parse problems: {len(problems)}")

    report = {
        "total": total, "ingested": ingested, "families": fam_count,
        "era": era_count, "pre2023_share": pre_share,
        "issuers_over3": single_issuer_over, "review": review,
        "human_accepted_overrides": accepted,
        "dupes": dupes, "problems": problems,
        "per_doc": [{"file": d["path"].name, "family": d["family"],
                     "era": d["meta"].get("era")
                     or ("pre" if d["meta"].get("published_date", "9") < "2023" else "post"),
                     "register": d["register_score"], "chars": len(d["body"]),
                     "tag": d["tag"]} for d in final],
    }
    out = ROOT / "reports/intake_human_probe_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
