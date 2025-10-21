# 🌐 Synq AI — Career & Event Companion

**Synq Auto** is an AI-powered career companion that helps users discover personalized tech events, career opportunities, and actionable insights.  
It combines **FastAPI**, **LangChain**, and **CrewAI** on the backend with a **modern Netlify frontend** — providing intelligent, real-time recommendations powered by LLMs.

---

## 🚀 Project Overview

Synq AI connects students, tech enthusiasts, and professionals to relevant **tech events in Nairobi and beyond**.  
Users can chat directly with **Synq Auto**, who tailors event suggestions, networking tips, and learning resources based on their interests, name, and schedule.

**Core Features**
- 🤖 Interactive AI chat assistant (Synq Auto)
- 🎯 Personalized event recommendations
- 🧭 Automated email insight system
- 🧩 FastAPI backend with LLM orchestration
- 🌍 Frontend hosted on Netlify
- ⚙️ Backend hosted on Render

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | HTML, CSS, JavaScript (Netlify) |
| **Backend** | FastAPI (Python) |
| **AI Orchestration** | LangChain, CrewAI |
| **Vector Store** | ChromaDB |
| **Embeddings** | Cohere / Gemini (configurable) |
| **Memory** | Redis or Chroma (optional) |
| **Deployment** | Render (Backend), Netlify (Frontend) |
| **Rate Limiting** | SlowAPI |
| **CORS Handling** | FastAPI Middleware |

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/TheuriEric/conturi.git
cd synq-ai
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # on macOS/Linux
venv\Scripts\activate      # on Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Backend Locally
```bash
uvicorn src.conturi.app:app --reload
```
Your API will be available at:  
👉 `http://127.0.0.1:8000`

---

## 🧑‍💻 Quick Start Guide

### Frontend (Netlify)
1. Deploy your frontend folder (HTML, JS, CSS) to Netlify.  
2. Set the backend API URLs in your JS:
   ```javascript
   const API_BASE = "https://synqai.onrender.com";
   ```

### Backend (Render)
1. Push your FastAPI code to GitHub.
2. Create a **new Render Web Service**.
3. Build command:
   ```bash
   pip install -r requirements.txt
   ```
4. Start command:
   ```bash
   uvicorn src.conturi.app:app --host 0.0.0.0 
   ```

---

## 🔐 Environment Variables

Create a `.env` file in your project root (not committed to Git).  
Example:

```bash
APP_NAME="Synq AI"
ENVIRONMENT="production"
COHERE_API_KEY="your-cohere-api-key"
GEMINI_API_KEY="your-gemini-api-key"
CHROMA_DB_PATH="./db/chroma.sqlite3"
REDIS_URL="redis://default:<password>@<host>:<port>"
NETLIFY_URL="https://synqio.netlify.app"
RENDER_URL="https://synqai.onrender.com"
SECRET_KEY="your-random-secret"
RATE_LIMIT="5/minute"
```

---

## 🔄 API Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/` | GET | Welcome message |
| `/chat` | POST | Chat endpoint for Synq Auto |
| `/automation` | POST | Sets up personalized email automation |

---

## 🧪 Example Request

```bash
curl -X POST https://synqai.onrender.com/chat   -H "Content-Type: application/json"   -d '{"query": "Find tech events in Nairobi this weekend"}'
```

**Example Response**
```json
{
  "response": "Here are 3 tech events happening in Nairobi this weekend..."
}
```

---

## 🛠️ Troubleshooting

| Issue | Cause | Fix |
|--------|--------|------|
| ⚠️ 500 Internal Server | Missing env vars | Check `.env` values on Render |
| 🚫 404 Not Found | Wrong endpoint URL | Verify `/chat` or `/automation` path |
| ⏱️ Rate limit exceeded | Too many requests | Adjust `RATE_LIMIT` in `.env` |

---

## 👥 Credits

Built by [**Eric Theuri**](https://github.com/TheuriEric/conturi.git)
> Empowering African students with smart, AI-driven career guidance and event discovery.

---

## 📄 License

This project is licensed under the **MIT License** — see `LICENSE` for details.
