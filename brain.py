"""Second Brain prototype: notes -> embeddings -> retrieval -> Groq answer, with caching.

Usage:
  python brain.py ingest
  python brain.py ask "your question" --role employee
  python brain.py chat --role founder
"""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path

# Fix Windows console Unicode encoding (cp1252 can't handle all LLM output)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chromadb
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
STORE_DIR = os.getenv("STORE_DIR", "store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
GROK_MODEL = os.getenv("GROK_MODEL", "llama-3.3-70b-versatile")  # Groq model
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.55"))      # above this = not relevant
CACHE_DISTANCE = float(os.getenv("CACHE_DISTANCE", "0.08"))  # below this = same question
CHUNK_CHARS = 900
PERSONA_FILE = DATA_DIR / "persona.md"

ROLE_ACCESS = {
    "employee": ["public"],
    "manager": ["public", "managers"],
    "founder": ["public", "managers", "founder-only"],
}

RULES = """You are an AI assistant that answers questions the way the founder would, using ONLY the numbered notes provided.
Rules:
- If the notes do not contain the answer, say: "That's not in the brain yet." Do not guess.
- Cite the notes you used like [1], [2].
- Never reveal or mention notes you were not given.
- You are an AI assistant, not the founder in person."""

_embedder = None


def embed(texts):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(texts, normalize_embeddings=True).tolist()


def db():
    client = chromadb.PersistentClient(path=STORE_DIR)
    cos = {"hnsw:space": "cosine"}
    return (client.get_or_create_collection("brain", metadata=cos),
            client.get_or_create_collection("qcache", metadata=cos))


# ---------- ingestion ----------
def parse_note(path):
    text = path.read_text(encoding="utf-8")
    meta = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            text = parts[2]
    defaults = {"company": "general", "department": "general", "access": "public", "type": "note", "date": ""}
    meta = {k: str(meta.get(k, v)) for k, v in defaults.items()}
    return meta, text


def chunk(body):
    sections, title, buf = [], "", []
    for line in body.splitlines():
        if re.match(r"#{1,3} ", line):
            if "".join(buf).strip():
                sections.append((title, "\n".join(buf).strip()))
            title, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((title, "\n".join(buf).strip()))
    out = []
    for title, text in sections:
        cur = ""
        for para in re.split(r"\n\s*\n", text):
            if cur and len(cur) + len(para) > CHUNK_CHARS:
                out.append((title, cur.strip()))
                cur = ""
            cur += para + "\n\n"
        if cur.strip():
            out.append((title, cur.strip()))
    return out


def ingest():
    brain, _ = db()
    existing = set(brain.get(include=[])["ids"])
    seen, ids, docs, metas = set(), [], [], []
    for path in sorted(DATA_DIR.rglob("*")):
        if path.suffix.lower() not in (".md", ".txt") or path == PERSONA_FILE:
            continue
        rel = str(path.relative_to(DATA_DIR))
        meta, body = parse_note(path)
        for title, text in chunk(body):
            doc = f"{title}\n{text}" if title else text
            cid = hashlib.sha1(f"{rel}|{doc}".encode()).hexdigest()
            seen.add(cid)
            if cid not in existing:  # embedding cache: unchanged chunks are never re-embedded
                ids.append(cid)
                docs.append(doc)
                metas.append({**meta, "source": rel, "title": title})
    if ids:
        brain.add(ids=ids, documents=docs, embeddings=embed(docs), metadatas=metas)
    stale = list(existing - seen)
    if stale:
        brain.delete(ids=stale)
    print(f"Ingested: {len(ids)} new, {len(seen) - len(ids)} reused, {len(stale)} removed")


# ---------- query ----------
def kb_version(brain):
    return hashlib.sha1("".join(sorted(brain.get(include=[])["ids"])).encode()).hexdigest()[:12]


def persona():
    extra = PERSONA_FILE.read_text(encoding="utf-8") if PERSONA_FILE.exists() else ""
    return f"{RULES}\n\nFounder persona:\n{extra}"  # stable prefix first, so the provider can reuse it


def answer(question, role="employee", company=None):
    t0 = time.time()
    brain, qcache = db()
    if brain.count() == 0:
        return {"answer": "The brain is empty. Run ingest first.", "sources": [], "cached": False, "ms": 0}

    qv = embed([question])[0]
    scope, kb = f"{role}|{company or '*'}", kb_version(brain)

    # 1) semantic cache: same scope, same knowledge version, near-identical question
    try:
        hit = qcache.query(query_embeddings=[qv], n_results=1,
                           where={"$and": [{"scope": scope}, {"kb": kb}]},
                           include=["documents", "metadatas", "distances"])
        if hit["ids"][0] and hit["distances"][0][0] <= CACHE_DISTANCE:
            return {"answer": hit["documents"][0][0],
                    "sources": json.loads(hit["metadatas"][0][0]["sources"]),
                    "cached": True, "ms": int((time.time() - t0) * 1000)}
    except Exception:
        pass

    # 2) retrieve, filtered by access BEFORE anything reaches the LLM
    where = {"access": {"$in": ROLE_ACCESS[role]}}
    if company:
        where = {"$and": [where, {"company": company}]}
    res = brain.query(query_embeddings=[qv], n_results=TOP_K, where=where,
                      include=["documents", "metadatas", "distances"])
    chunks = [(d, m) for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
              if dist <= MAX_DISTANCE]

    if not chunks:  # nothing relevant: skip the LLM call entirely
        text, sources = "That's not in the brain yet.", []
    else:
        ctx = "\n\n".join(f"[{i + 1}] ({m['source']}) {d}" for i, (d, m) in enumerate(chunks))
        client = OpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=GROK_MODEL, temperature=0.3,
            messages=[{"role": "system", "content": persona()},
                      {"role": "user", "content": f"Notes:\n{ctx}\n\nQuestion: {question}"}])
        text = resp.choices[0].message.content
        sources = sorted({m["source"] for _, m in chunks})

    qcache.add(ids=[hashlib.sha1(f"{scope}|{kb}|{question}".encode()).hexdigest()],
               embeddings=[qv], documents=[text],
               metadatas=[{"scope": scope, "kb": kb, "question": question, "sources": json.dumps(sources)}])
    return {"answer": text, "sources": sources, "cached": False, "ms": int((time.time() - t0) * 1000)}


def show(r):
    print(f"\n{r['answer']}\n")
    if r["sources"]:
        print("Sources:", ", ".join(r["sources"]))
    print(f"[{'cache hit' if r['cached'] else 'fresh'} | {r['ms']} ms]\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ingest", "ask", "chat"])
    p.add_argument("question", nargs="?")
    p.add_argument("--role", default="employee", choices=list(ROLE_ACCESS))
    p.add_argument("--company")
    a = p.parse_args()
    if a.cmd == "ingest":
        ingest()
    elif a.cmd == "ask":
        show(answer(a.question, a.role, a.company))
    else:
        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit", ""):
                break
            show(answer(q, a.role, a.company))
