import os
import requests
import pickle
import threading
import time
import logging
from threading import Lock
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient

# -------------------- Config --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "tiiuae/falcon-7b-instruct")  # CPU-friendly
TOP_K = int(os.getenv("TOP_K", 3))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 3000))
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 300))  # seconds

if not SUPABASE_URL or not SUPABASE_KEY or not HF_TOKEN:
    raise RuntimeError("SUPABASE_URL, SUPABASE_KEY, HF_TOKEN must be set.")

lock = Lock()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------- FastAPI --------------------
app = FastAPI(title="Posts AI Q&A Service")

class Question(BaseModel):
    text: str

# -------------------- Globals --------------------
embed_model = SentenceTransformer(EMBEDDING_MODEL)
index = None
texts = []

# -------------------- Fetch posts --------------------
def fetch_posts(user_id: Optional[str] = None):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }
    try:
        posts_resp = requests.get(f"{SUPABASE_URL}/rest/v1/posts", headers=headers)
        posts_resp.raise_for_status()
        posts = {p['id']: p for p in posts_resp.json()}

        merged_texts = []

        if user_id:
            saved_resp = requests.get(f"{SUPABASE_URL}/rest/v1/saved_posts?user_id=eq.{user_id}", headers=headers)
            saved_resp.raise_for_status()
            saved_posts = saved_resp.json()
            for sp in saved_posts:
                post = posts.get(sp['post_id'])
                if post and post.get("content"):
                    merged_texts.append(f"{post.get('page_name','')} {post.get('content','')} {post.get('permalink','')}")

        # fallback to all posts
        if not merged_texts:
            merged_texts = [
                f"{p.get('page_name','')} {p.get('content','')} {p.get('permalink','')}"
                for p in posts.values() if p.get("content")
            ]

        return merged_texts

    except Exception as e:
        logging.error(f"Error fetching posts: {e}")
        return []

# -------------------- FAISS index --------------------
def build_index(texts_list):
    embeddings = embed_model.encode(texts_list, convert_to_numpy=True)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    with open("texts.pkl", "wb") as f:
        pickle.dump(texts_list, f)
    faiss.write_index(idx, "posts.index")
    logging.info(f"FAISS index built with {len(texts_list)} entries")
    return idx

def refresh_index():
    global index, texts
    while True:
        try:
            new_texts = fetch_posts()
            with lock:
                if set(new_texts) != set(texts):
                    texts = new_texts
                    index = build_index(texts)
                    logging.info("FAISS index updated dynamically")
        except Exception as e:
            logging.error(f"Error refreshing index: {e}")
        time.sleep(REFRESH_INTERVAL)

# -------------------- Clean text --------------------
def clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

# -------------------- Q&A --------------------
def answer_question(question_text: str):
    try:
        query_vec = embed_model.encode([question_text])
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        with lock:
            D, I = index.search(query_vec, k=TOP_K)
            relevant_context = [clean_text(texts[i][:MAX_CONTEXT_CHARS]) for i in I[0]]

        context_text = "\n".join(f"- {ctx}" for ctx in relevant_context)
        prompt = f"Answer the question using these posts:\n{context_text}\n\nQuestion: {question_text}\nAnswer:"

        client = InferenceClient(token=HF_TOKEN, model=GENERATION_MODEL)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat_completion(
            messages,
            max_tokens=512,
            stream=False,
            temperature=0.7,
            top_p=0.95
        )

        answer = "".join(
            choice.message.content for choice in response.choices if choice.message and choice.message.content
        )
        return clean_text(answer), relevant_context

    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "Error generating answer", []

# -------------------- FastAPI routes --------------------
@app.get("/")
def root():
    return {"message": "Service running"}

@app.post("/ask")
def ask(q: Question):
    answer, context = answer_question(q.text)
    return {"answer": answer, "context": context}

# -------------------- Startup event --------------------
@app.on_event("startup")
def startup_event():
    global index, texts
    # Load index if exists
    if os.path.exists("posts.index") and os.path.exists("texts.pkl"):
        index = faiss.read_index("posts.index")
        with open("texts.pkl", "rb") as f:
            texts = pickle.load(f)
    else:
        texts = fetch_posts()
        index = build_index(texts)

    # Start background refresh thread
    threading.Thread(target=refresh_index, daemon=True).start()
    logging.info("Startup complete, FAISS index ready.")

# -------------------- Run --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
