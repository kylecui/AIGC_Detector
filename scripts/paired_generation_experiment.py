"""Paired-generation experiment: free-form vs contract-constrained prompts.

W4 of .sisyphus/plans/fn1-countermeasures-and-paired-experiment.md
Hypothesis: contract-constrained generation (arm B) yields lower detector p_ai
than free-form (arm A) — but only for models whose contract-adherence
capability is above the structural floor (capability-gate refinement, see
DETECTOR_NOTES_2026-08.md W4 pilot).

Capability ladder (validated in kylecui/contract-driven-harness-study):
- Qwen/Qwen2.5-7B-Instruct (local, 4-bit)  — below-floor candidate (0/40)
- THUDM/GLM-4-9B-0414        (SiliconFlow)  — above floor (30/40), FN-1 family
- deepseek-ai/DeepSeek-V3.2  (SiliconFlow)  — above floor (40/40)

Design: N topics x arms {A free-form, B output-contract} x models.
v1.1: B-arm spec enforces >=800 total chars (length-confound fix vs pilot).

Usage:
    uv run python scripts/paired_generation_experiment.py --backend api --n-topics 40
    uv run python scripts/paired_generation_experiment.py --backend local --n-topics 40
    (checkpointed resume by model|topic|arm)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUT_DIR = Path("dataset/paired_generation_v1")
RECORDS = OUT_DIR / "pilot_records.jsonl"
LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
API_BASE = "https://api.siliconflow.cn/v1"
API_MODELS = ["THUDM/GLM-4-9B-0414", "deepseek-ai/DeepSeek-V3.2"]

GEN_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 1600,  # local
    "api_max_tokens": 3000,  # API arm, v1.1 length floor needs headroom
    "repetition_penalty": 1.1,
}

TOPICS = [
    {"id": "t01", "doc_type": "数据安全合规承诺书", "subject": "某科技公司向监管部门提交数据处理合规承诺", "org": "上海某科技有限公司"},
    {"id": "t02", "doc_type": "情况说明", "subject": "就线上服务中断四小时向客户说明原因与处置", "org": "某互联网服务公司"},
    {"id": "t03", "doc_type": "道歉声明", "subject": "就产品批次质量问题向消费者公开致歉", "org": "某消费品公司"},
    {"id": "t04", "doc_type": "授权声明", "subject": "授权关联公司在约定范围内使用注册商标", "org": "某品牌管理公司"},
    {"id": "t05", "doc_type": "保密承诺书", "subject": "新入职员工就接触商业秘密作出保密承诺", "org": "某研发型企业"},
    {"id": "t06", "doc_type": "廉洁自律承诺书", "subject": "采购岗位员工承诺拒绝商业贿赂与利益输送", "org": "某制造企业"},
    {"id": "t07", "doc_type": "整改报告", "subject": "就安全生产检查发现问题报告整改措施与时限", "org": "某工程施工单位"},
    {"id": "t08", "doc_type": "免责声明", "subject": "群众性体育活动组织方就参与风险作出提示", "org": "某体育赛事公司"},
    {"id": "t09", "doc_type": "地址变更公告", "subject": "公司办公地址搬迁并说明业务衔接安排", "org": "某金融服务公司"},
    {"id": "t10", "doc_type": "质量保证承诺书", "subject": "就工程项目施工质量作出保修期承诺", "org": "某建设集团"},
    {"id": "t11", "doc_type": "复工申请书", "subject": "停工整改完成后向住建部门申请恢复施工", "org": "某建筑工程公司"},
    {"id": "t12", "doc_type": "延期情况说明", "subject": "就项目交付延期两个月向甲方作出解释", "org": "某软件外包公司"},
    {"id": "t13", "doc_type": "供应商承诺函", "subject": "投标供应商就货期与售后服务作出承诺", "org": "某机电设备公司"},
    {"id": "t14", "doc_type": "退款说明函", "subject": "就课程服务无法履约向学员说明退款安排", "org": "某教育培训机构"},
    {"id": "t15", "doc_type": "环保承诺书", "subject": "生产企业就污染物排放达标向生态部门承诺", "org": "某化工企业"},
    {"id": "t16", "doc_type": "志愿服务承诺书", "subject": "大型展会志愿者就服务纪律与保密作出承诺", "org": "某会展服务公司"},
    {"id": "t17", "doc_type": "资金监管协议履行声明", "subject": "就预售资金监管账户使用情况向购房人声明", "org": "某房地产开发公司"},
    {"id": "t18", "doc_type": "产品召回公告", "subject": "就某型号儿童玩具安全隐患启动召回", "org": "某玩具制造公司"},
    {"id": "t19", "doc_type": "离职交接确认书", "subject": "离职员工就工作与文件移交完毕作出确认", "org": "某咨询公司"},
    {"id": "t20", "doc_type": "安全责任承诺书", "subject": "施工单位就动火作业安全责任作出承诺", "org": "某安装工程公司"},
    {"id": "t21", "doc_type": "个人信息保护承诺书", "subject": "App运营者就用户个人信息处理规则作出承诺", "org": "某移动互联网公司"},
    {"id": "t22", "doc_type": "更正声明", "subject": "就更正此前公告中错误数据向社会公开说明", "org": "某上市公司"},
    {"id": "t23", "doc_type": "和解协议履行确认函", "subject": "就交通事故赔偿协议履行完毕予以确认", "org": "某保险公司"},
    {"id": "t24", "doc_type": "高校实验室安全承诺书", "subject": "研究生就实验操作规范与责任作出承诺", "org": "某高校实验室"},
    {"id": "t25", "doc_type": "食品安全自查报告", "subject": "连锁餐饮门店就季度食品安全自查结果报告", "org": "某餐饮管理公司"},
    {"id": "t26", "doc_type": "终止合作公告", "subject": "就终止某代言人合作并向公众说明", "org": "某消费品品牌公司"},
    {"id": "t27", "doc_type": "投标澄清函", "subject": "就招标方对标书疑点提出的问题作出澄清", "org": "某系统集成商"},
    {"id": "t28", "doc_type": "员工竞业限制确认书", "subject": "核心员工离职时就竞业限制义务作出确认", "org": "某人工智能公司"},
    {"id": "t29", "doc_type": "灾害损失情况报告", "subject": "就台风造成仓储损失向保险机构报告", "org": "某物流公司"},
    {"id": "t30", "doc_type": "招生简章免责条款说明", "subject": "民办学校就招生宣传边界向主管部门说明", "org": "某民办学校"},
    {"id": "t31", "doc_type": "版权声明", "subject": "就平台原创内容版权归属与转载规则作出声明", "org": "某内容平台公司"},
    {"id": "t32", "doc_type": "设备验收异议函", "subject": "就采购设备验收不合格向供应商提出异议", "org": "某制造企业"},
    {"id": "t33", "doc_type": "电力设施保护承诺书", "subject": "施工单位就邻近高压线路作业作出安全承诺", "org": "某市政工程公司"},
    {"id": "t34", "doc_type": "患者知情同意履行说明", "subject": "医院就特殊治疗知情同意流程作出说明", "org": "某三甲医院"},
    {"id": "t35", "doc_type": "社区物业费调价公告", "subject": "物业公司就物业费调整向业主公告", "org": "某物业服务公司"},
    {"id": "t36", "doc_type": "进出口合规声明", "subject": "外贸企业就两用物项出口合规作出声明", "org": "某进出口贸易公司"},
    {"id": "t37", "doc_type": "学术诚信承诺书", "subject": "考生就考试独立完成与材料真实作出承诺", "org": "某认证考试机构"},
    {"id": "t38", "doc_type": "临时占用道路申请附函", "subject": "就占道施工围挡安排向交管部门作出说明", "org": "某燃气工程公司"},
    {"id": "t39", "doc_type": "系统升级停机公告", "subject": "银行就核心系统升级窗口向客户公告", "org": "某商业银行"},
    {"id": "t40", "doc_type": "股权变更情况告知函", "subject": "公司就股东股权变更向债权人告知", "org": "某实业投资公司"},
]


def freeform_prompt(t: dict) -> str:
    return (
        f"请写一份{t['doc_type']}，背景：{t['subject']}，以{t['org']}名义。"
        "要求内容完整、语言正式规范，800字左右。直接输出正文。"
    )


def contract_prompt(t: dict) -> str:
    return f"""请严格按照以下规格撰写一份{t['doc_type']}，背景：{t['subject']}，出具单位：{t['org']}。

【输出结构——按以下顺序，各项缺一不可】
1. 标题：《关于……的{t['doc_type']}》
2. 称谓行：以"致："开头，写明接收方
3. 总起段：一段话说明出具目的，以"兹因"开头
4. 基本信息表：紧随总起段，以"项目：内容"逐行列出至少4行（如 出具单位/适用范围/生效日期/联系方式）
5. 正文条款：以"一、二、三、"编号，共4条，每条须以一个主题词开头（如"一、承诺事项："）
6. 结尾段：以"特此承诺。"或"特此声明。"单独成段
7. 落款两行：出具单位名称 + "＿＿＿＿年＿＿月＿＿日"

【语言约束】
- 使用规范公文体；必须出现下列法定套语中的至少3处："依据""兹因""郑重""严格遵守""如有违反""承担相应责任"
- 不得使用口语、网络用语、感叹号；不得出现解释性或元话语（如"以下是我写的"）

【长度约束】全文总字数（含表格与落款）不少于800字、不超过1000字；单段不超过150字。

【格式禁止】不使用任何Markdown标记（#、*、-）、不添加小标题、直接输出正文。"""


SYSTEM_PROMPT = "你是一名熟悉中国公文写作规范的文书撰写助手。"


# ---------------- local backend ----------------

def load_local_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL, quantization_config=qcfg, device_map="cuda"
    )
    model.eval()
    return model, tok


def gen_local(model, tok, user_prompt: str) -> tuple[str, float]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=GEN_PARAMS["max_new_tokens"],
            temperature=GEN_PARAMS["temperature"],
            top_p=GEN_PARAMS["top_p"],
            repetition_penalty=GEN_PARAMS["repetition_penalty"],
            do_sample=True,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip(), time.time() - t0


# ---------------- API backend ----------------

def gen_api(api_key: str, model_id: str, user_prompt: str) -> tuple[str, float]:
    import httpx

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": GEN_PARAMS["temperature"],
        "top_p": GEN_PARAMS["top_p"],
        "max_tokens": GEN_PARAMS["api_max_tokens"],
        "stream": False,
    }
    t0 = time.time()
    last_err = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=420) as client:
                r = client.post(
                    f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"].strip()
                if "</think>" in text:  # strip stray reasoning blocks
                    text = text.split("</think>", 1)[1].strip()
                return text, time.time() - t0
        except Exception as e:  # noqa: BLE001 — retry any transport/API error
            last_err = e
            time.sleep(5 * (2**attempt))
    raise RuntimeError(f"API failed after 3 retries: {last_err}")


# ---------------- orchestration ----------------

def done_keys() -> set[str]:
    keys: set[str] = set()
    if RECORDS.exists():
        for line in RECORDS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add(f"{r['model']}|{r['topic_id']}|{r['arm']}")
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "api", "all"], default="api")
    ap.add_argument("--n-topics", type=int, default=40)
    ap.add_argument("--budget-seconds", type=int, default=500)
    ap.add_argument("--topics-file", type=str, default=None,
                    help="JSON array of topic dicts (id/doc_type/subject/org); "
                         "replaces built-in TOPICS")
    ap.add_argument("--spec-tag", type=str, default=None,
                    help="spec_version tag for records; required with --topics-file "
                         "(default v1.2-replication)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.topics_file:
        topics = json.loads(Path(args.topics_file).read_text(encoding="utf-8"))
        spec_tag = args.spec_tag or "v1.2-replication"
    else:
        topics = TOPICS[: args.n_topics]
        spec_tag = "v1.1-lengthfloor"

    jobs: list[tuple[str, dict, str]] = []  # (model, topic, arm)
    dk = done_keys()
    if args.backend in ("local", "all"):
        jobs += [(LOCAL_MODEL, t, a) for t in topics for a in ("A", "B")
                 if f"{LOCAL_MODEL}|{t['id']}|{a}" not in dk]
    if args.backend in ("api", "all"):
        for m in API_MODELS:
            jobs += [(m, t, a) for t in topics for a in ("A", "B")
                     if f"{m}|{t['id']}|{a}" not in dk]
    if not jobs:
        print("nothing to do")
        return 0
    print(f"pending: {len(jobs)} generations")

    local_model = local_tok = None
    if any(m == LOCAL_MODEL for m, _, _ in jobs):
        local_model, local_tok = load_local_model()
        print("local model loaded")

    api_key = ""
    if any(m != LOCAL_MODEL for m, _, _ in jobs):
        import os

        from aigc_detector.config import settings

        api_key = os.environ.get("SILICONFLOW_API_KEY") or settings.openai_api_key or ""
        if not api_key:
            print("ERROR: API models requested but no SILICONFLOW_API_KEY / openai_api_key")
            return 1

    t_start = time.time()
    n_done = 0
    with RECORDS.open("a", encoding="utf-8") as fh:
        for model_id, t, arm in jobs:
            if time.time() - t_start > args.budget_seconds:
                print(f"budget exhausted after {n_done}; re-run to resume")
                break
            prompt = freeform_prompt(t) if arm == "A" else contract_prompt(t)
            if model_id == LOCAL_MODEL:
                text, elapsed = gen_local(local_model, local_tok, prompt)
            else:
                text, elapsed = gen_api(api_key, model_id, prompt)
            if len(text) < 200:
                print(f"  WARN short {model_id.split('/')[-1]} {t['id']}/{arm}: {len(text)} chars")
            rec = {
                "id": hashlib.sha1(
                    f"{t['id']}|{arm}|{model_id}".encode()
                ).hexdigest()[:10],
                "topic_id": t["id"],
                "doc_type": t["doc_type"],
                "arm": arm,
                "model": model_id,
                "backend": "local" if model_id == LOCAL_MODEL else "siliconflow",
                "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10],
                "text": text,
                "char_len": len(text),
                "gen_seconds": round(elapsed, 1),
                "gen_params": GEN_PARAMS,
                "spec_version": spec_tag,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            print(f"  {model_id.split('/')[-1]} {t['id']}/{arm} {len(text)}c {elapsed:.0f}s")

    meta = {
        "experiment": "paired-generation freeform-vs-contract",
        "spec_version": spec_tag,
        "capability_ladder": {
            LOCAL_MODEL: "below-floor candidate (harness study 0/40)",
            "THUDM/GLM-4-9B-0414": "above floor (30/40), FN-1 family",
            "deepseek-ai/DeepSeek-V3.2": "above floor (40/40)",
        },
        "gen_params": GEN_PARAMS,
        "n_topics": len(topics),
        "arms": {"A": "free-form prompt", "B": "output-contract spec v1.1"},
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"done: {n_done} new records -> {RECORDS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
