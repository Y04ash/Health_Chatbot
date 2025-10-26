import os
import pickle
import asyncio
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import httpx  # Import async-capable HTTP client
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from pydantic import Field

# Initialize Flask App
app = Flask(__name__)
load_dotenv()

# --- Environment Variables ---
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
print(f"GROQ API Key Loaded: {GROQ_API_KEY is not None}")
print(f"Pinecone API Key Loaded: {PINECONE_API_KEY is not None}")
# --- 1. Embeddings Caching ---
# Define a cache file path
EMBEDDINGS_CACHE_FILE = "embeddings_cache.pkl"

def load_or_create_embeddings():
    """
    Loads embeddings from a local cache if it exists, otherwise downloads
    and caches them for future use.
    """
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        print("Loading embeddings from cache...")
        with open(EMBEDDINGS_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    else:
        print("Downloading and caching embeddings...")
        embeddings = download_hugging_face_embeddings()
        with open(EMBEDDINGS_CACHE_FILE, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

# Load embeddings using the caching mechanism
embeddings = load_or_create_embeddings()

# --- Pinecone and LangChain Setup ---
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# --- 2. Asynchronous LLM Class ---
class GroqChatLLM(LLM):
    """
    Custom LLM class for Groq API that supports both synchronous and
    asynchronous calls.
    """
    api_key: str = Field(..., description="Groq API key")
    model: str = Field("meta-llama/llama-4-scout-17b-16e-instruct", description="Model name")
    api_url: str = "https://api.groq.com/openai/v1/chat/completions"

    # Synchronous call method (for compatibility)
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # This will now be a wrapper around the async call
        return asyncio.run(self._acall(prompt, stop))

    # Asynchronous call method
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "groq-chat"

# --- RAG Chain Initialization ---
llm = GroqChatLLM(api_key=GROQ_API_KEY)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
async def chat():  # Make the function asynchronous with 'async'
    msg = request.form["msg"]
    # Use the asynchronous 'ainvoke' method and 'await' the result
    response = await rag_chain.ainvoke({"input": msg})
    return str(response["answer"])

# Note: The 'if __name__ == "__main__":' block is removed.
# You should run this app using an ASGI server like Hypercorn.