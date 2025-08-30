import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# -------------------- Config --------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-oss-20b")
TOP_K = int(os.getenv("TOP_K", 3))

if not DATABASE_URL or not HF_TOKEN:
    raise RuntimeError("DATABASE_URL and HF_TOKEN must be set in environment variables.")

# -------------------- Database --------------------
engine = create_engine(DATABASE_URL)

# -------------------- Embeddings --------------------
if os.path.exists("posts.index") and os.path.exists("texts.pkl"):
    index = faiss.read_index("posts.index")
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)
else:
    # Fetch and merge data
    posts_df = pd.read_sql(text("SELECT * FROM posts"), engine)
    saved_df = pd.read_sql(text("SELECT * FROM saved_posts"), engine)
    merged_df = saved_df.merge(posts_df, left_on='post_id', right_on='id', how='left')
    
    texts = (
        merged_df['page_name'].fillna('') + " " +
        merged_df['content'].fillna('') + " " +
        merged_df['permalink'].fillna('')
    ).tolist()
    
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    faiss.write_index(index, "posts.index")
    with open("texts.pkl", "wb") as f:
        pickle.dump(texts, f)

# Load embedding model
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
        relevant_context = [texts[i] for i in I[0]]
        
        # 3. Build prompt
        context_text = "\n".join(relevant_context)
        prompt = f"Answer the question based on these posts:\n{context_text}\n\nQuestion: {q.text}\nAnswer:"
        
        # 4. Call Hugging Face GPT Inference
        client = InferenceClient(token=HF_TOKEN, model=GENERATION_MODEL)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": prompt})
        
        answer = ""
        for message in client.chat_completion(messages, max_tokens=512, stream=True):
            choices = message.choices
            if len(choices) and choices[0].delta.content:
                answer += choices[0].delta.content
        
        return {"answer": answer.strip(), "context": relevant_context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Posts AI Q&A service running"}
    