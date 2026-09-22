import os
import json
import tempfile
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

from parsing import parse_file, chunk_text
from vectorstore import add_chunks, query as vector_query, list_documents, new_doc_id, delete_document
from agent_tools import TOOLS, run_tool
from analytics import load_dataframe, analyze_dataframe, analytics_to_text_summary
from transcription import transcribe_audio
from pdf_reports import build_document_summary_pdf, build_analytics_report_pdf, build_transcript_report_pdf, build_generic_text_pdf
from auth import register_user, login_user

# In-memory store of analytics results, keyed by filename (fine for a hackathon demo;
# use a real DB if you need persistence across restarts)
ANALYTICS_STORE: dict[str, dict] = {}

# In-memory store of transcripts, keyed by a generated name — holds the raw text,
# timestamped segments (if from an audio file), and the generated summary.
TRANSCRIPT_STORE: dict[str, dict] = {}

# In-memory store of full document text, keyed by filename — needed to regenerate
# a professional summary for the PDF export without re-parsing the file.
DOCUMENT_TEXT_STORE: dict[str, str] = {}

app = FastAPI(title="Scribe — AI Notes, Transcripts & Documents (Free Stack)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# NOTE: llama-3.3-70b-versatile is now Enterprise-only on Groq (not on the free tier).
# openai/gpt-oss-120b is free-tier with real rate limits (250K TPM / 1K RPM) and supports tool calling.
MODEL = "openai/gpt-oss-120b"


class QueryRequest(BaseModel):
    question: str


DATA_EXTENSIONS = {"csv", "xlsx", "xls"}
DOC_EXTENSIONS = {"pdf", "docx", "txt"}


AUDIO_VIDEO_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "mov", "webm", "ogg"}


class SummarizeTextRequest(BaseModel):
    text: str
    title: str = "Live Transcript"


class ExportTextRequest(BaseModel):
    title: str
    text: str


class AuthRequest(BaseModel):
    email: str
    password: str
    role: str = "user"
    admin_code: str | None = None


def summarize_transcript(text: str) -> dict:
    """
    Shared summarization logic for both recorded and live transcripts.
    Generates FOUR distinct outputs:
    - brief: a 1-2 sentence quick overview
    - summary: a general, flowing-paragraph summary
    - mom: formal Minutes of Meeting (key points, decisions, action items)
    - sentiment: overall tone + explanation
    """
    system_prompt = (
        "You are an expert meeting/interview summarizer. Given a raw transcript, "
        "produce a JSON object with EXACTLY these four keys and nothing else:\n"
        '"brief": a 1-2 sentence quick overview of what this recording is about.\n'
        '"summary": a general summary written in flowing paragraphs (not bullet points), '
        "covering the overall content, context, and tone of the discussion.\n"
        '"mom": formal Minutes of Meeting, written in markdown with these exact sections: '
        "**Key Points** (bullet list), **Decisions Made** (bullet list, say 'None identified' "
        "if none), **Action Items** (bullet list with an owner name if mentioned, say "
        "'None identified' if none).\n"
        '"sentiment": an object with two keys: "overall" (one of: Positive, Neutral, Negative, '
        "Mixed) and \"explanation\" (2-3 sentences on the emotional tone, engagement level, "
        "and any notable shifts in tone during the conversation).\n"
        "Base every field ONLY on the transcript provided — never invent details. "
        "Respond with ONLY the raw JSON object — no markdown code fences, no extra commentary."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript:\n\n{text}"},
        ],
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()

    # Be defensive: strip ```json fences if the model adds them anyway.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        # Fallback: if JSON parsing fails, put everything in "summary" so nothing is lost.
        parsed = {"brief": "", "summary": raw, "mom": "", "sentiment": {}}

    return {
        "brief": parsed.get("brief", ""),
        "summary": parsed.get("summary", ""),
        "mom": parsed.get("mom", ""),
        "sentiment": parsed.get("sentiment", {"overall": "Unknown", "explanation": ""}),
    }


def summarize_document_professional(filename: str, text: str) -> str:
    """
    Generates a professional, well-structured summary of a full document
    (not just RAG chunks) — used for the downloadable PDF summary.
    """
    system_prompt = (
        "You are a professional business analyst. Given a document's full text, write a "
        "clear, well-organized summary suitable for an executive report. Structure it with: "
        "1) Overview (2-3 sentences), 2) Key Points (bullet list, prefix each with '- '), "
        "3) Notable Details (bullet list of specific facts, figures, or dates worth flagging, "
        "prefix each with '- '). Base this ONLY on the provided text — never invent details. "
        "Write in clear, professional prose."
    )
    # Guard against extremely long documents exceeding context — truncate generously.
    truncated = text[:15000]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document: {filename}\n\n{truncated}"},
        ],
        max_tokens=1800,
    )
    return response.choices[0].message.content.strip()





@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    """
    Upload a recorded audio/video file -> transcribe locally (free, no API cost)
    -> generate meeting-minutes-style summary -> embed transcript for RAG chat.
    """
    ext = file.filename.lower().rsplit(".", 1)[-1]
    if ext not in AUDIO_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio/video type: {ext}")

    content = await file.read()
    # faster-whisper reads from a file path, so we write to a temp file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = transcribe_audio(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        os.remove(tmp_path)

    summary_result = summarize_transcript(result["text"])

    # Store for retrieval by the frontend, and embed for RAG chat.
    TRANSCRIPT_STORE[file.filename] = {
        "text": result["text"],
        "segments": result["segments"],
        "brief": summary_result["brief"],
        "summary": summary_result["summary"],
        "mom": summary_result["mom"],
        "sentiment": summary_result["sentiment"],
        "duration": result["duration"],
        "language": result["language"],
    }

    chunks = chunk_text(result["text"], chunk_size=1200, overlap=200)
    doc_id = new_doc_id()
    count = add_chunks(doc_id, file.filename, chunks)

    return {
        "filename": file.filename,
        "type": "transcript",
        "duration": result["duration"],
        "language": result["language"],
        "brief": summary_result["brief"],
        "summary": summary_result["summary"],
        "mom": summary_result["mom"],
        "sentiment": summary_result["sentiment"],
        "chunks_stored": count,
    }


@app.post("/summarize-live")
def summarize_live(req: SummarizeTextRequest):
    """
    For LIVE transcription: the browser's built-in speech recognition (free,
    no server cost) does the real-time transcribing. When the user stops
    recording, the full text is sent here to be summarized and embedded,
    same as a recorded file.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No transcript text provided")

    summary_result = summarize_transcript(req.text)

    TRANSCRIPT_STORE[req.title] = {
        "text": req.text,
        "segments": [],
        "brief": summary_result["brief"],
        "summary": summary_result["summary"],
        "mom": summary_result["mom"],
        "sentiment": summary_result["sentiment"],
        "duration": None,
        "language": "live",
    }

    chunks = chunk_text(req.text, chunk_size=1200, overlap=200)
    doc_id = new_doc_id()
    count = add_chunks(doc_id, req.title, chunks)

    return {
        "filename": req.title,
        "type": "transcript",
        "brief": summary_result["brief"],
        "summary": summary_result["summary"],
        "mom": summary_result["mom"],
        "sentiment": summary_result["sentiment"],
        "chunks_stored": count,
    }


@app.get("/transcripts")
def list_transcripts():
    return {"files": list(TRANSCRIPT_STORE.keys())}


@app.get("/transcripts/{filename}")
def get_transcript(filename: str):
    transcript = TRANSCRIPT_STORE.get(filename)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return transcript


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Handles two kinds of uploads:
    - Documents (pdf/docx/txt) -> parsed, chunked, embedded for RAG chat
    - Data files (csv/xlsx) -> analyzed for stats/charts, AND summarized + embedded
      so you can also ask questions about the data in chat
    """
    content = await file.read()
    ext = file.filename.lower().rsplit(".", 1)[-1]

    if ext in DATA_EXTENSIONS:
        try:
            df = load_dataframe(file.filename, content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read data file: {e}")

        analytics = analyze_dataframe(df)
        ANALYTICS_STORE[file.filename] = analytics

        # Also embed a text summary of the data so it's queryable in chat
        summary_text = analytics_to_text_summary(file.filename, analytics)
        chunks = chunk_text(summary_text, chunk_size=1500, overlap=100)
        doc_id = new_doc_id()
        count = add_chunks(doc_id, file.filename, chunks)

        return {
            "filename": file.filename,
            "type": "data",
            "doc_id": doc_id,
            "chunks_stored": count,
            "rows": analytics["shape"]["rows"],
            "columns": analytics["shape"]["columns"],
        }

    if ext in DOC_EXTENSIONS:
        try:
            text = parse_file(file.filename, content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        DOCUMENT_TEXT_STORE[file.filename] = text

        chunks = chunk_text(text)
        doc_id = new_doc_id()
        count = add_chunks(doc_id, file.filename, chunks)

        return {"filename": file.filename, "type": "document", "doc_id": doc_id, "chunks_stored": count}

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


@app.get("/analytics/{filename}")
def get_analytics(filename: str):
    """Return full analytics (stats, missing values, correlation, chart data) for a data file."""
    analytics = ANALYTICS_STORE.get(filename)
    if not analytics:
        raise HTTPException(status_code=404, detail="No analytics found for this file")
    return analytics


@app.get("/analytics")
def list_analytics():
    """List all data files that have analytics available."""
    return {"files": list(ANALYTICS_STORE.keys())}


@app.get("/analytics-summary/all")
def get_summary_board():
    """
    Aggregate overview across every uploaded dataset — powers the Summary Board.
    Combines row/column totals, an averaged quality score, and the top insight
    per file so the dashboard can show a single cross-file snapshot.
    """
    files = list(ANALYTICS_STORE.items())
    if not files:
        return {
            "total_datasets": 0,
            "total_rows": 0,
            "total_columns": 0,
            "avg_quality_score": None,
            "files": [],
        }

    total_rows = sum(a["shape"]["rows"] for _, a in files)
    total_columns = sum(a["shape"]["columns"] for _, a in files)
    scores = [a["summary_board"]["quality_score"] for _, a in files]
    avg_quality_score = round(sum(scores) / len(scores)) if scores else None

    file_summaries = [
        {
            "filename": fname,
            "rows": a["shape"]["rows"],
            "columns": a["shape"]["columns"],
            "quality_score": a["summary_board"]["quality_score"],
            "top_insight": (a["summary_board"]["insights"][0] if a["summary_board"]["insights"] else None),
        }
        for fname, a in files
    ]

    return {
        "total_datasets": len(files),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "avg_quality_score": avg_quality_score,
        "files": file_summaries,
    }


@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


@app.delete("/documents/{filename}")
def delete_document_endpoint(filename: str):
    """Delete a file's chunks from the vector store, and its analytics if it was a data file."""
    deleted_count = delete_document(filename)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="File not found")

    # Also remove analytics/transcript data if this filename was one of those
    ANALYTICS_STORE.pop(filename, None)
    TRANSCRIPT_STORE.pop(filename, None)

    return {"filename": filename, "chunks_deleted": deleted_count}


@app.post("/query")
def query_documents(req: QueryRequest):
    """
    RAG endpoint:
    1. Retrieve relevant chunks
    2. Let the LLM decide to answer directly or call a tool
    3. Return the answer WITH citations
    """
    hits = vector_query(req.question, n_results=6)

    if not hits:
        return {
            "answer": "I don't have any relevant documents to answer that. Please upload a document first.",
            "citations": [],
        }

    context_text = "\n\n---\n\n".join(
        f"[Source: {h['filename']}, chunk {h['chunk_index']}]\n{h['text']}" for h in hits
    )

    system_prompt = (
        "You are a thorough, expert document analyst. Answer using ONLY the provided context. "
        "If the answer isn't in the context, say so clearly — never guess or hallucinate. "
        "Give complete, well-explained answers: include relevant details, numbers, and reasoning "
        "from the context rather than a one-line summary. Where useful, structure your answer with "
        "short paragraphs or bullet points for clarity. "
        "Always state which source file(s) support your answer. "
        "If the user's request matches one of your tools (summarize, extract dates, flag risks), use it."
    )

    user_message = f"Context:\n{context_text}\n\nQuestion: {req.question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        max_tokens=2000,
    )

    choice = response.choices[0].message

    if choice.tool_calls:
        tool_call = choice.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments or "{}")
        tool_result_text = run_tool(tool_name, tool_args, context_text)

        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {"name": tool_name, "arguments": tool_call.function.arguments},
            }
        ]})
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result_text,
        })

        follow_up = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=2000,
        )
        answer_text = follow_up.choices[0].message.content
    else:
        answer_text = choice.content

    citations = [{"filename": h["filename"], "chunk_index": h["chunk_index"]} for h in hits]
    return {"answer": answer_text, "citations": citations}


@app.post("/export/text-pdf")
def export_text_pdf(req: ExportTextRequest):
    """Export any block of text (typically a chat answer) as a formatted PDF."""
    pdf_bytes = build_generic_text_pdf(req.title, req.text)
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in req.title)[:60] or "export"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.pdf"'},
    )


@app.post("/auth/register")
def auth_register(req: AuthRequest):
    try:
        user = register_user(req.email, req.password, req.role, req.admin_code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "Account created", "user": user}


@app.post("/auth/login")
def auth_login(req: AuthRequest):
    try:
        user = login_user(req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return {"user": user}


@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# PDF REPORT DOWNLOADS
# ============================================================

@app.get("/documents/{filename}/summary-pdf")
def download_document_summary_pdf(filename: str):
    """Generate and download a professional PDF summary of a document."""
    text = DOCUMENT_TEXT_STORE.get(filename)
    if not text:
        raise HTTPException(status_code=404, detail="Document text not found — try re-uploading it")

    summary_text = summarize_document_professional(filename, text)
    pdf_bytes = build_document_summary_pdf(filename, summary_text)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}-summary.pdf"'},
    )


@app.get("/analytics/{filename}/report-pdf")
def download_analytics_report_pdf(filename: str):
    """Generate and download a full end-to-end analytics report (charts, tables, insights) as PDF."""
    analytics = ANALYTICS_STORE.get(filename)
    if not analytics:
        raise HTTPException(status_code=404, detail="Analytics not found for this file")

    pdf_bytes = build_analytics_report_pdf(filename, analytics)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}-analytics-report.pdf"'},
    )


@app.get("/transcripts/{filename}/report-pdf")
def download_transcript_report_pdf(filename: str):
    """Generate and download a full meeting/interview report (brief, summary, MOM, sentiment) as PDF."""
    transcript = TRANSCRIPT_STORE.get(filename)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    pdf_bytes = build_transcript_report_pdf(filename, transcript)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}-report.pdf"'},
    )