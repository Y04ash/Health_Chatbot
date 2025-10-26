

# 🧠 Medical Chatbot using LangChain, Pinecone, and Groq LLM

A powerful AI-powered chatbot designed to answer medical queries using real-time vector search and LLM-based responses. Built using **Flask**, **LangChain**, **Pinecone**, and **Groq API**, and deployed for public access via **Render**.

## 📌 Live Demo

🔗 [Try the Chatbot Here](https://healthai-amla.onrender.com)  


---

## 💡 Project Overview

This chatbot answers user queries based on a medical knowledge base using a Retrieval-Augmented Generation (RAG) architecture. It combines document similarity search via **Pinecone** and intelligent response generation using **Groq’s LLaMA model**.

---

## 🔧 Features

✅ Real-time chatbot interface using HTML, CSS, and JavaScript  
✅ Server-side logic using Flask and Python  
✅ RAG pipeline using LangChain: combines retrieval and LLM response  
✅ Uses **Pinecone vector DB** for semantic document search  
✅ Uses **Groq’s hosted LLaMA 4 model** via API for fast and intelligent replies  
✅ Responsive and styled chat UI  
✅ Fully deployed on **Render** with a public link

---

## 🛠️ Tech Stack

| Category       | Technology                     |
|----------------|--------------------------------|
| Frontend       | HTML, CSS (Bootstrap, jQuery)  |
| Backend        | Python, Flask                  |
| LLM            | Groq API (LLaMA 4)             |
| Vector DB      | Pinecone                       |
| Retrieval/RAG  | LangChain                      |
| Deployment     | Render                         |
| Environment    | `.env` for API keys            |

---

## 🧠 Architecture

```

User ➝ Flask App ➝ LangChain ➝
➝ Pinecone (Retrieval)
➝ Groq API (LLM Response)
➝ Final Answer ➝ Chat UI

```

---

## 🧪 How I Set Up Pinecone

1. Created a **Pinecone account** and a new index named `medicalbot`
2. Used Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` to generate vector embeddings
3. Uploaded text/document chunks to the Pinecone index
4. Integrated Pinecone with LangChain's `PineconeVectorStore` to retrieve top 3 relevant documents for any user query

---

## 🤖 How I Integrated Groq LLM

1. Generated a **Groq API key**
2. Created a custom `GroqChatLLM` class extending `langchain.llms.base.LLM`
3. Used Groq’s hosted LLaMA 4 Scout model via REST API
4. Combined with LangChain’s prompt and retriever in a `RetrievalChain`

---

## 📁 Folder Structure

```

├── app.py                # Main Flask app
├── requirements.txt      # All dependencies
├── templates/
│   └── chat.html         # Chatbot frontend
├── static/
│   └── style.css         # Styling
├── .env                  # Pinecone and Groq API keys (not pushed)
├── src/
│   └── helper.py         # Embedding utility

````

---

## 🚀 Deployment on Render

1. Pushed project to GitHub
2. Connected GitHub repo to Render
3. Set the `Start Command` as: `gunicorn app:app`
4. Set environment variables for:
   - `PINECONE_API_KEY`
   - `GROQ_API_KEY`
5. Render builds and exposes a public URL

---

## 📌 Environment Variables (`.env`)

```env
PINECONE_API_KEY=your_pinecone_key_here
GROQ_API_KEY=your_groq_key_here
````

*(These are set securely in Render's dashboard and not pushed to GitHub)*

---

## 🧠 Future Improvements

* Upload PDF documents to index dynamically
* Add user authentication
* Expand to other domains (e.g., legal, education)
* Analytics dashboard for user questions

---

## 🙋‍♂️ About Me

I’m IT student passionate about AI and real-world applications of LLMs. This project reflects my hands-on learning of AI pipelines, LangChain, and cloud deployment. Open to internships and AI-focused opportunities.

---

## 📬 Contact

📧 [atharvnikam38@gmail.com](mailto:atharvnikam38@gmail.com)
🐙 [GitHub](https://github.com/atharvnikam38)

---

> If you like this project, feel free to ⭐ the repo!





