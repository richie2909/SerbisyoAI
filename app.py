import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from huggingface_hub import InferenceClient
import threading
import time

# -------------------- Config --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")  # e.g., https://xyz.supabase.co/rest/v1
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-oss-20b")
TOP_K = int(os.getenv("TOP_K", 3))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 3000))
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 300))  # seconds

if not SUPABASE_URL or not SUPABASE_KEY or not HF_TOKEN:
    raise RuntimeError("SUPABASE_URL, SUPABASE_KEY, and HF_TOKEN must be set.")

# -------------------- Fetch posts from Supabase --------------------
def fetch_posts():
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }
    posts_resp = requests.get(f"{SUPABASE_URL}/rest/v1/posts", headers=headers)
    saved_resp = requests.get(f"{SUPABASE_URL}/rest/v1/saved_posts", headers=headers)
    posts_resp.raise_for_status()
    saved_resp.raise_for_status()

    posts = {p['id']: p for p in posts_resp.json()}
    saved_posts = saved_resp.json()

    merged_texts = []
    for sp in saved_posts:
        post = posts.get(sp['post_id'], {})
        merged_text = f"{post.get('page_name','')} {post.get('content','')} {post.get('permalink','')}"
        merged_texts.append(merged_text)

    return merged_texts

# -------------------- Embeddings & FAISS --------------------
from threading import Lock
lock = Lock()

def build_index(texts):
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    faiss.write_index(idx, "posts.index")
    with open("texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    return idx

def refresh_index_periodically():
    global index, texts
    while True:
        try:
            new_texts = fetch_posts()
            with lock:
                if set(new_texts) != set(texts):
                    texts = new_texts
                    index = build_index(texts)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAISS index updated dynamically.")
        except Exception as e:
            print(f"Error refreshing index: {e}")
        time.sleep(REFRESH_INTERVAL)

# -------------------- Initialize --------------------
embed_model = SentenceTransformer(EMBEDDING_MODEL)

if os.path.exists("posts.index") and os.path.exists("texts.pkl"):
    index = faiss.read_index("posts.index")
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)
else:
    texts = fetch_posts()
    index = build_index(texts)

# Start dynamic refresh in background
threading.Thread(target=refresh_index_periodically, daemon=True).start()

# -------------------- FastAPI --------------------
app = FastAPI(title="Posts AI Q&A Service")

class Question(BaseModel):
    text: str

@app.post("/ask")
def ask_question(q: Question):
    try:
        query_vec = embed_model.encode([q.text])
        with lock:
            D, I = index.search(query_vec, k=TOP_K)
            relevant_context = [texts[i][:MAX_CONTEXT_CHARS] for i in I[0]]

        context_text = "\n".join(relevant_context)
        prompt = f"Answer the question based on these posts:\n{context_text}\n\nQuestion: {q.text}\nAnswer:"

        client = InferenceClient(token=HF_TOKEN, model=GENERATION_MODEL)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": prompt})

        response = client.chat_completion(
            messages,
            max_tokens=512,
            stream=False,
            temperature=0.7,
            top_p=0.95
        )

        answer = ""
        for choice in response.choices:
            if choice.message and choice.message.content:
                answer += choice.message.content

        return {"answer": answer.strip(), "context": relevant_context}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Posts AI Q&A service running"}
