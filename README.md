<<<<<<< HEAD
# Scribe — AI Notes, Transcripts & Documents

A unified, 100% free AI system combining document intelligence (RAG), data analytics, and audio/video transcription + summarization.

## What it does
- **Documents** (PDF/DOCX/TXT) → RAG chat with citations
- **Datasets** (CSV/XLSX) → automatic analytics, quality score, insights, charts
- **Transcripts** → upload a recording OR record live in-browser → free local transcription → AI-generated meeting minutes (overview, key points, decisions, action items) → also chattable via RAG

Everything — documents, datasets, and transcripts — lives in one shared chat, so you can ask questions across all of it.

## Free stack
| Layer | Tool | Cost |
|---|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) | Free tier |
| Embeddings | sentence-transformers (local) | Free |
| Vector DB | ChromaDB (local) | Free |
| Transcription | faster-whisper (local, offline after first run) | Free |
| Live transcription | Browser's Web Speech API (Chrome/Edge) | Free |
| Analytics | pandas | Free |
| Charts | Chart.js | Free |

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # add your free Groq key from console.groq.com
uvicorn main:app --reload --port 8000
```
First transcription run downloads the Whisper "base" model (~150MB) once — needs internet that one time, fully offline after.

### Frontend
```bash
cd frontend
python -m http.server 3000
```
Open `http://localhost:3000`

## Live transcription notes
- Uses the browser's built-in Web Speech API — **Chrome or Edge only**, no server cost, no API key needed
- Click "Start live recording", speak, click "Stop & summarize" — the transcript is sent to the backend for AI summarization and embedding
- For recorded files instead, use "Upload recording" — supports mp3, wav, m4a, mp4, mov, webm, ogg

## Architecture
- `parsing.py` — document text extraction + chunking
- `analytics.py` — dataset analysis, quality score, insights
- `transcription.py` — faster-whisper wrapper for audio/video
- `vectorstore.py` — embeddings + ChromaDB
- `agent_tools.py` — RAG chat tool-calling (summarize, extract dates, flag risks)
- `main.py` — FastAPI app tying it all together
- `frontend/index.html` — single-file React UI (Documents / Datasets / Transcripts tabs + shared chat)

## Deployment
Same free path as before: Render (backend, free tier — note faster-whisper needs more RAM than the free tier's 512MB may comfortably allow for larger files; keep test audio short, or upgrade if needed) + Vercel (frontend).
=======
# scribe
>>>>>>> beaccf86b56940cd96fd8cdc186d89c9da358ca1
