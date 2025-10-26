import os
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import requests
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from pydantic import Field

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')



embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

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

class GroqChatLLM(LLM):
    api_key: str = Field(..., description="Groq API key")
    model: str = Field("meta-llama/llama-4-scout-17b-16e-instruct", description="Model name")
    api_url: str = "https://api.groq.com/openai/v1/chat/completions"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
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
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "groq-chat"

llm = GroqChatLLM(api_key=GROQ_API_KEY)
print("GROQ_API_KEY Loaded:", GROQ_API_KEY)
print("Pinecone_API_KEY Loaded:", PINECONE_API_KEY)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
