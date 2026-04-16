# ⚖️ NomoSys – AI Legal Chatbot

NomoSys is an AI-powered legal assistant designed to simplify Indian law and make legal knowledge accessible to everyone. It allows users to ask legal questions in plain language and receive clear, structured, and reference-backed answers.

The project demonstrates two powerful approaches:

* 🌐 **API-Based Model (MERN + HuggingFace)**
* 🧠 **RAG-Based Local LLM (Ollama + DeepSeek + FAISS)**

---

## 🚀 Features

* 🗣️ Natural language legal query support
* 📚 Covers Indian Constitution, IPC, CrPC, Company Act, etc.
* 🔍 Retrieval-Augmented Generation (RAG) for accurate answers
* 🌐 Cloud-based chatbot (globally accessible)
* 💻 Offline chatbot (no internet required)
* 📄 Document-based legal query support
* 📊 Explainable responses with legal references

---

## 🏗️ Architecture

### 1️⃣ API-Based Approach (MERN + HuggingFace)

User → React UI → Node/Express Backend → HuggingFace API → Response

* Fast and scalable
* Deployed using **Vercel (Frontend)** + **Render (Backend)**
* Depends on internet connectivity

---

### 2️⃣ RAG-Based Approach (Ollama + DeepSeek + FAISS)

User → Streamlit UI → Embedding → FAISS Retrieval → DeepSeek LLM → Response

* High accuracy (context-based answers)
* Works offline
* Requires high hardware resources

---

## 🛠️ Tech Stack

### 🌐 API Version

* MongoDB
* Express.js
* React.js
* Node.js
* HuggingFace Inference API
* Vercel + Render

### 🧠 RAG Version

* Python
* Ollama
* DeepSeek 671B
* FAISS Vector Database
* Streamlit

---

## ⚙️ Installation & Setup

### 🔹 API-Based Version

```bash
# Clone repo
git clone https://github.com/your-username/nyayasathi.git

# Backend
cd backend
npm install
npm start

# Frontend
cd frontend
npm install
npm start
```

> Add your HuggingFace API key in `.env`

---

### 🔹 RAG-Based Version

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

> Make sure **Ollama** and **DeepSeek model** are installed locally.

---

## 🧪 Test Cases

* ✔️ Valid legal query (e.g., Article 21)
* ✔️ Random input handling
* ✔️ Non-legal query rejection
* ✔️ Complex multi-law scenarios
* ✔️ Document-based query

---

## 📊 Results

| Model     | Performance                            |
| --------- | -------------------------------------- |
| API Model | Fast, scalable, moderate accuracy      |
| RAG Model | High accuracy, offline, resource-heavy |

---

## ⚠️ Limitations

* RAG model requires high-end hardware (GPU recommended)
* Cannot deploy large models on Vercel/Render
* API model depends on internet and rate limits
* Not a substitute for professional legal advice

---

## 🔮 Future Enhancements

* 🎤 Voice-based legal queries
* 🌍 Multilingual support
* 📱 Mobile app
* ⚖️ Case-law retrieval system
* 🧠 Explainable AI reasoning layer

---

## 🎯 Use Cases

* Legal awareness for citizens
* Educational tool for students
* NGO and legal aid support
* Rural legal assistance

---

## 👨‍💻 Contributors

* Karthik Maiya

---

## ⭐ Acknowledgements

* HuggingFace
* Ollama
* DeepSeek
* FAISS
* Government of India (Legal Documents)

---

## ⚠️ Disclaimer

NomoSys provides **legal information only**, not legal advice.
For official matters, please consult a qualified legal professional.

---
