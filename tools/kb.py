#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb.py —— 破晓/戴世 项目知识库工具（零外部依赖，纯 Python 3）

把 deals/ 下所有 Markdown（分析、原文提取、问题清单、纪要、跟踪）建成
一个可全文检索的知识库，并自动生成 INDEX.md 看板。

为什么是 BM25 / 词法检索而不是神经向量检索？
  本仓库的远程执行环境网络白名单不放行 huggingface.co，且无 embeddings
  API key，无法下载/调用向量模型。BM25 + 中文分词在 BP 这种术语密集语料上
  召回很好，且零依赖、可在任意全新容器复现。
  如需升级为语义向量检索：实现下方 EmbeddingBackend，设置好 API key 即可，
  search() 的调用方式不变（见文件末尾说明）。

用法：
  python3 tools/kb.py build              # 扫描 deals/ 重建索引
  python3 tools/kb.py search "触觉数据壁垒"   # 全文检索，返回 谁/哪页/原文
  python3 tools/kb.py search "估值" -n 8 --project 戴世   # 限定项目/条数
  python3 tools/kb.py index              # 重新生成根目录 INDEX.md 看板
  python3 tools/kb.py all                # build + index 一把梭
"""
import os, re, json, math, sys, argparse, glob
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEALS = os.path.join(ROOT, "deals")
INDEX_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".kb_index.json")
INDEX_MD = os.path.join(ROOT, "INDEX.md")

CJK = r"一-鿿㐀-䶿"

# ----------------------------- 分词 -----------------------------
def tokenize(text):
    """中文按 字 + 二元组 切，英文/数字按词切。无需 jieba。"""
    text = text.lower()
    toks = []
    # 拉丁字母 / 数字 / 含小数点百分号的词（如 99%, 4000, 1000hz, 1°/h, sim2real）
    for m in re.findall(r"[a-z0-9][a-z0-9._%/+-]*", text):
        toks.append(m.strip("._-/"))
    # CJK：单字 + 相邻二元组
    for run in re.findall(f"[{CJK}]+", text):
        chars = list(run)
        toks.extend(chars)
        toks.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return [t for t in toks if t]

# ------------------------- frontmatter 解析 -------------------------
def parse_frontmatter(raw):
    """极简 YAML 子集解析：key: value，列表写成 [a, b, c]。"""
    meta = {}
    if not raw.startswith("---"):
        return meta, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return meta, raw
    block = raw[3:end].strip("\n")
    body = raw[end + 4:]
    for line in block.splitlines():
        if not line.strip() or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            v = [x.strip() for x in v[1:-1].split(",") if x.strip()]
        meta[k] = v
    return meta, body

# ----------------------------- 切块 -----------------------------
def chunk_markdown(body):
    """按 Markdown 标题切块，标题作为该块的定位锚。"""
    chunks, cur_head, buf = [], "(开头)", []
    for line in body.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if "".join(buf).strip():
                chunks.append((cur_head, "\n".join(buf).strip()))
            cur_head = line.lstrip("#").strip()
            buf = []
        else:
            buf.append(line)
    if "".join(buf).strip():
        chunks.append((cur_head, "\n".join(buf).strip()))
    return chunks

def project_of(path):
    rel = os.path.relpath(path, DEALS)
    return rel.split(os.sep)[0]

# ----------------------------- 建索引 -----------------------------
def build():
    docs, df = [], defaultdict(int)
    for path in sorted(glob.glob(os.path.join(DEALS, "**", "*.md"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        _, body = parse_frontmatter(raw)
        rel = os.path.relpath(path, ROOT)
        proj = project_of(path)
        for head, text in chunk_markdown(body):
            if not text.strip():
                continue
            toks = tokenize(head + "。" + text)
            if not toks:
                continue
            tf = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t in tf:
                df[t] += 1
            docs.append({
                "project": proj, "file": rel, "head": head,
                "text": text, "len": len(toks), "tf": dict(tf),
            })
    avgdl = sum(d["len"] for d in docs) / len(docs) if docs else 0
    json.dump({"docs": docs, "df": dict(df), "N": len(docs), "avgdl": avgdl},
              open(INDEX_JSON, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"✅ 已索引 {len(docs)} 个文本块，来自 {len(set(d['file'] for d in docs))} 个文件 → {os.path.relpath(INDEX_JSON, ROOT)}")

# ----------------------------- BM25 检索 -----------------------------
def search(query, topn=6, project=None, k1=1.5, b=0.75):
    if not os.path.exists(INDEX_JSON):
        print("（索引不存在，自动重建…）"); build()
    idx = json.load(open(INDEX_JSON, encoding="utf-8"))
    docs, df, N, avgdl = idx["docs"], idx["df"], idx["N"], idx["avgdl"]
    q = tokenize(query)
    scored = []
    for d in docs:
        if project and project not in d["project"]:
            continue
        s = 0.0
        for t in q:
            if t not in d["tf"]:
                continue
            n = df.get(t, 0)
            idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
            f = d["tf"][t]
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * d["len"] / avgdl))
        if s > 0:
            scored.append((s, d))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        print(f"未命中：「{query}」"); return
    qset = set(q)
    print(f"\n🔎 「{query}」 命中 {len(scored)} 块，Top {min(topn, len(scored))}：\n")
    for rank, (s, d) in enumerate(scored[:topn], 1):
        snippet = _snippet(d["text"], qset)
        print(f"[{rank}] {d['project']}  ·  {d['head']}   (score {s:.1f})")
        print(f"    📄 {d['file']}")
        print(f"    {snippet}\n")

def _snippet(text, qset, width=140):
    """取命中词附近的一段，去掉换行。"""
    flat = re.sub(r"\s+", " ", text)
    low = flat.lower()
    pos = -1
    for t in sorted(qset, key=len, reverse=True):
        p = low.find(t)
        if p != -1:
            pos = p; break
    if pos == -1:
        return flat[:width] + ("…" if len(flat) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(flat), start + width)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")

# ----------------------------- 生成 INDEX.md 看板 -----------------------------
def build_index_md():
    rows = []
    for ana in sorted(glob.glob(os.path.join(DEALS, "*", "01-分析.md"))):
        meta, _ = parse_frontmatter(open(ana, encoding="utf-8").read())
        proj = project_of(ana)
        products = sorted(glob.glob(os.path.join(DEALS, proj, "0*.md")))
        stages = "".join(
            {"01": "①", "02": "②", "03": "③", "04": "④"}.get(os.path.basename(p)[:2], "")
            for p in products)
        tags = meta.get("标签", [])
        tags = " ".join(f"`{t}`" for t in (tags if isinstance(tags, list) else [tags]))[:60]
        rows.append({
            "公司": meta.get("公司", proj),
            "赛道": meta.get("赛道", "-"),
            "轮次": meta.get("轮次", "-"),
            "融资额": meta.get("融资额", "-"),
            "评级": meta.get("评级", ""),
            "结论": meta.get("结论", "-"),
            "进度": stages or "①",
            "更新": meta.get("更新", "-"),
            "标签": tags,
            "dir": proj,
        })
    rows.sort(key=lambda r: r["更新"], reverse=True)
    lines = [
        "# 项目知识库 · 总索引看板", "",
        "> 本文件由 `python3 tools/kb.py index` 自动生成，请勿手改。",
        "> 全文检索：`python3 tools/kb.py search \"关键词\"`", "",
        f"**在投项目：{len(rows)}**　进度图例：① 分析 ② 问题 ③ 纪要 ④ 跟踪", "",
        "| 公司 | 赛道 | 轮次 | 融资额 | 评级 | 结论 | 进度 | 更新 |",
        "|------|------|------|--------|------|------|------|------|",
    ]
    for r in rows:
        link = f"[{r['公司']}](deals/{r['dir']}/01-分析.md)"
        lines.append(f"| {link} | {r['赛道']} | {r['轮次']} | {r['融资额']} | {r['评级']} | {r['结论']} | {r['进度']} | {r['更新']} |")
    lines += ["", "## 标签", ""]
    for r in rows:
        lines.append(f"- **{r['公司']}**：{r['标签']}")
    lines += ["", "---", "", "### 怎么用这个知识库", "",
              "```bash", "python3 tools/kb.py build              # 改完内容后重建索引",
              "python3 tools/kb.py search \"触觉数据壁垒\"   # 搜：谁在哪页说了什么",
              "python3 tools/kb.py search \"估值\" --project 戴世", "python3 tools/kb.py index               # 重新生成本看板", "```", ""]
    open(INDEX_MD, "w", encoding="utf-8").write("\n".join(lines))
    print(f"✅ 已生成 {os.path.relpath(INDEX_MD, ROOT)}（{len(rows)} 个项目）")

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="项目知识库：建索引 / 全文检索 / 生成看板")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("build")
    sp = sub.add_parser("search")
    sp.add_argument("query")
    sp.add_argument("-n", type=int, default=6)
    sp.add_argument("--project", default=None)
    sub.add_parser("index")
    sub.add_parser("all")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "search":
        search(a.query, a.n, a.project)
    elif a.cmd == "index":
        build_index_md()
    elif a.cmd == "all":
        build(); build_index_md()
    else:
        ap.print_help()

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 升级到语义向量检索（当环境放行 embeddings 服务时）：
#   1) 实现 def embed(texts:list[str]) -> list[list[float]]，调用 Voyage/OpenAI。
#   2) build() 里给每个 chunk 存 d["vec"]=embed([d["text"]])[0]。
#   3) search() 里改为 query 向量与各 chunk 向量做余弦相似度排序。
#   词法 BM25 可与向量分数加权融合（hybrid），通常召回更稳。
# ---------------------------------------------------------------------------
