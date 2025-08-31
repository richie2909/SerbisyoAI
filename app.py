import os
import requests
import pickle
import threading
import time
import logging
from threading import Lock
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# -------------------- Load Environment --------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
TOP_K = int(os.getenv("TOP_K", 4))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 3000))
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 300))

if not SUPABASE_URL or not SUPABASE_KEY or not HF_TOKEN:
    raise RuntimeError("❌ Missing SUPABASE_URL, SUPABASE_KEY, or HF_TOKEN. Check your .env file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info("✅ Environment loaded: SUPABASE_URL=%s, EMBEDDING_MODEL=%s, GENERATION_MODEL=%s",
             SUPABASE_URL, EMBEDDING_MODEL, GENERATION_MODEL)

# -------------------- FastAPI --------------------
app = FastAPI(title="Posts AI Q&A Service")

class Question(BaseModel):
    text: str

class AnswerResponse(BaseModel):
    answer: str
    context: List[str]

# -------------------- Globals --------------------
embed_model = SentenceTransformer(EMBEDDING_MODEL)
index = None
texts: List[str] = []
lock = Lock()

# -------------------- Fetch posts --------------------
def fetch_posts(user_id: Optional[str] = None):
    """Fetch posts from Supabase and embed timestamps into the context for better time awareness."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }
    try:
        posts_resp = requests.get(f"{SUPABASE_URL}/rest/v1/posts", headers=headers, timeout=10)
        posts_resp.raise_for_status()
        posts = {p['id']: p for p in posts_resp.json()}

        merged_texts = []
        if user_id:
            try:
                saved_resp = requests.get(
                    f"{SUPABASE_URL}/rest/v1/saved_posts?user_id=eq.{user_id}",
                    headers=headers,
                    timeout=10
                )
                saved_resp.raise_for_status()
                saved_posts = saved_resp.json()
                for sp in saved_posts:
                    post = posts.get(sp['post_id'])
                    if post and post.get("content"):
                        merged_texts.append(
                            f"[{post.get('page_name','Unknown Page')} | Created: {post.get('created_at','N/A')} | "
                            f"Last Synced: {post.get('last_synced','N/A')}] {post.get('content','')} "
                            f"(Link: {post.get('permalink','')})"
                        )
            except Exception as e:
                logging.warning(f"⚠️ Could not fetch saved_posts for user {user_id}: {e}")

        if not merged_texts:
            merged_texts = [
                f"[{p.get('page_name','Unknown Page')} | Created: {p.get('created_at','N/A')} | "
                f"Last Synced: {p.get('last_synced','N/A')}] {p.get('content','')} "
                f"(Link: {p.get('permalink','')})"
                for p in posts.values() if p.get("content")
            ]
        return merged_texts

    except Exception as e:
        logging.error(f"❌ Error fetching posts: {e}")
        return []

# -------------------- FAISS index --------------------
def build_index(texts_list):
    """Build FAISS index from list of texts."""
    if not texts_list:
        logging.warning("⚠️ No texts to index. Skipping index build.")
        return None
    try:
        embeddings = embed_model.encode(texts_list, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        dim = embeddings.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(embeddings)
        with open("texts.pkl", "wb") as f:
            pickle.dump(texts_list, f)
        faiss.write_index(idx, "posts.index")
        logging.info(f"✅ FAISS index built with {len(texts_list)} entries")
        return idx
    except Exception as e:
        logging.error(f"❌ Failed to build index: {e}")
        return None

def refresh_index():
    """Background thread to refresh FAISS index periodically."""
    global index, texts
    while True:
        try:
            new_texts = fetch_posts()
            with lock:
                if set(new_texts) != set(texts):
                    texts = new_texts
                    index = build_index(texts)
                    logging.info("♻️ FAISS index refreshed dynamically")
        except Exception as e:
            logging.error(f"Error refreshing index: {e}")
        time.sleep(REFRESH_INTERVAL)

# -------------------- Clean text --------------------
def clean_text(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())

# -------------------- Q&A --------------------
def answer_question(question_text: str):
    """Answer questions concisely using context from indexed posts."""
    try:
        query_vec = embed_model.encode([question_text], convert_to_numpy=True, normalize_embeddings=True)
        query_vec = query_vec.astype("float32")

        with lock:
            if index is None:
                return "⚠️ No indexed posts yet. Please try again shortly.", []
            D, I = index.search(query_vec, k=TOP_K)
            relevant_context = [clean_text(texts[i][:MAX_CONTEXT_CHARS]) for i in I[0] if i < len(texts)]

        if not relevant_context:
            return "I couldn't find any relevant information in the posts.", []

        context_text = "\n".join(f"- {ctx}" for ctx in relevant_context)
        prompt = (
            f"Use ONLY the context below to answer the question. Be concise, clear, and factual. "
            f"If no relevant information is found, answer exactly: 'I couldn't find any information about that.'\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question_text}\nAnswer:"
        )

        client = InferenceClient(GENERATION_MODEL, token=HF_TOKEN)
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "Provide short, clear, and direct answers. Avoid long explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.2,  # even lower = very deterministic
            top_p=0.9
        )

        answer = "".join(
            choice.message.content for choice in response.choices if choice.message and choice.message.content
        ).strip()
        return clean_text(answer), relevant_context

    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return f"❌ Error generating answer: {e}", []

# -------------------- FastAPI routes --------------------
@app.get("/")
def root():
    return {"message": "✅ Service running"}

@app.post("/ask", response_model=AnswerResponse)
def ask(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    answer, context = answer_question(q.text)
    return AnswerResponse(answer=answer, context=context)

# -------------------- Startup event --------------------
@app.on_event("startup")
def startup_event():
    global index, texts
    try:
        if os.path.exists("posts.index") and os.path.exists("texts.pkl"):
            index = faiss.read_index("posts.index")
            with open("texts.pkl", "rb") as f:
                texts = pickle.load(f)
            logging.info("✅ Loaded existing FAISS index and texts.")
        else:
            texts = fetch_posts()
            index = build_index(texts)
        threading.Thread(target=refresh_index, daemon=True).start()
        logging.info("🚀 Startup complete, FAISS index ready.")
    except Exception as e:
        logging.error(f"❌ Startup failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "true").lower() == "true"
    )
