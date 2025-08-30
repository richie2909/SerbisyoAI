import gradio as gr
import pandas as pd
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# -------------------- Config & DB --------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# -------------------- Embeddings --------------------
if os.path.exists("posts.index") and os.path.exists("texts.pkl"):
    index = faiss.read_index("posts.index")
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)
else:
    # Fetch data
    posts_df = pd.read_sql(text("SELECT * FROM posts"), engine)
    saved_df = pd.read_sql(text("SELECT * FROM saved_posts"), engine)
    merged_df = saved_df.merge(posts_df, left_on='post_id', right_on='id', how='left')
    
    # Combine for context
    texts = (merged_df['page_name'].fillna('') + " " + 
             merged_df['content'].fillna('') + " " + 
             merged_df['permalink'].fillna('')).tolist()
    
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, "posts.index")
    with open("texts.pkl", "wb") as f:
        pickle.dump(texts, f)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------- Chat Function --------------------
def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    """
    Respond using posts context + Hugging Face GPT model.
    """
    # 1. Retrieve top 3 relevant posts
    query_vec = embed_model.encode([message])
    D, I = index.search(query_vec, k=3)
    relevant_context = [texts[i] for i in I[0]]
    
    # 2. Build system + context prompt
    prompt = f"{system_message}\n\nRelevant posts:\n"
    prompt += "\n".join(relevant_context)
    prompt += f"\n\nUser question: {message}\nAnswer:"

    # 3. Call Hugging Face Inference API
    client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        choices = message.choices
        token = ""
        if len(choices) and choices[0].delta.content:
            token = choices[0].delta.content
        response += token
        yield response

# -------------------- Gradio Chat --------------------
chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()

if __name__ == "__main__":
    demo.launch()
