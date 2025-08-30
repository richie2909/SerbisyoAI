import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from huggingface_hub import InferenceClient

# -------------------- Config --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")  # e.g., https://xyz.supabase.co/rest/v1
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-oss-20b")
TOP_K = int(os.getenv("TOP_K", 3))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 3000))

if not SUPABASE_URL or not SUPABASE_KEY or not HF_TOKEN:
    raise RuntimeError("SUPABASE_URL, SUPABASE_KEY, and HF_TOKEN must be set.")

# -------------------- Fetch posts from Supabase --------------------
def fetch_posts():
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    # Fetch saved_posts joined with posts
    posts_resp = requests.get(f"{SUPABASE_URL}/posts", headers=headers)
    saved_resp = requests.get(f"{SUPABASE_URL}/saved_posts", headers=headers)
    posts_resp.raise_for_status()
    saved_resp.raise_for_status()

    posts = {p['id']: p for p in posts_resp.json()}
    saved_posts = saved_resp.json()

    # Merge saved_posts with posts
    merged_texts = []
    for sp in saved_posts:
        post = posts.get(sp['post_id'], {})
        merged_text = f"{post.get('page_name','')} {post.get('content','')} {post.get('permalink','')}"
        merged_texts.append(merged_text)
    return merged_texts

# -------------------- Embeddings --------------------
if os.path.exists("posts.index") and os.path.exists("texts.pkl"):
    index = faiss.read_index("posts.index")
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)
else:
    texts = fetch_posts()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_model.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "posts.index")
    with open("texts.pkl", "wb") as f:
        pickle.dump(texts, f)

# Load embedding model once
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# -------------------- FastAPI Setup --------------------
app = FastAPI(title="Posts AI Q&A Service")

class Question(BaseModel):
    text: str

@app.post("/ask")
def ask_question(q: Question):
    try:
        # 1. Embed question
        query_vec = embed_model.encode([q.text])

        # 2. Search FAISS index
        D, I = index.search(query_vec, k=TOP_K)
        relevant_context = [texts[i][:MAX_CONTEXT_CHARS] for i in I[0]]  # Trim context

        # 3. Build prompt
        context_text = "\n".join(relevant_context)
        prompt = f"Answer the question based on these posts:\n{context_text}\n\nQuestion: {q.text}\nAnswer:"

        # 4. Call Hugging Face GPT Inference
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
