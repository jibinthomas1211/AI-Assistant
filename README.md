# 🧠 Study Assistant

**AI-powered personal knowledge assistant** built entirely in **Python**.
It helps you store, search, and query your study materials — including **PDFs, Word files, PowerPoints, text files, and images** — using **vector search** and an **LLM**.

---

## 🚀 Features

* 📂 **Ingests multiple file types** — PDF, DOCX, PPTX, TXT, Images
* 🧩 **Extracts and chunks text automatically**
* 🧠 **Embeds and stores data in Pinecone** for semantic search
* 💬 **Answers natural-language questions** using your uploaded materials
* 🖥️ **Streamlit interface** for managing uploads and queries

---

## ⚙️ Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **Pinecone**
* **OpenAI API**
* **SentenceTransformers**

Supports both **OpenAI** and **SBERT** embeddings.

---

## 🏃‍♂️ Getting Started

### 1️⃣ Clone the repository and install dependencies

```bash
git clone https://github.com/jibinthomas1211/AI-Assistant.git
cd AI-Assistant
pip install -r requirements.txt
```

### 2️⃣ Set environment variables

```bash
export OPENAI_API_KEY=your_key
export PINECONE_API_KEY=your_key
export PINECONE_ENV=your_region
```

### 3️⃣ Run the app

```bash
streamlit run assistant.py
```

---

✅ **Now upload your study materials and start chatting with your notes!**

---
