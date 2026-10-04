# AUTOSAR HLD Document Analysis Assistant

An intelligent, local-first engineering assistant for automotive software architects and systems engineers. Built according to **Case Study 1: AUTOSAR HLD Document Analysis Assistant** from the Automotive Engineering AI Project Standard.

This application automates the ingestion, structural decomposition, knowledge extraction, grounded question-answering (RAG), and consistency verification of large AUTOSAR High-Level Design (HLD) specifications—including Software Components (SWCs), Interfaces, Ports, Signals, and Dependencies.

---

## 🏛️ Architectural Overview

```
                      ┌────────────────────────────────────────┐
                      │    Streamlit Pilot UI (Port 8501)      │
                      │  - Document Ingestion & Metrics        │
                      │  - Grounded Chat & Dual Citations      │
                      │  - Architecture Inventory (SWC/Port)   │
                      │  - Inconsistency Inspector & Audit Log │
                      └──────────────────┬─────────────────────┘
                                         │ REST HTTP
                                         ▼
                      ┌────────────────────────────────────────┐
                      │      FastAPI Backend (Port 8000)       │
                      │  - Ingestion, Parsing & Chunking Engine│
                      │  - Strict Grounding & Citation Verifier│
                      │  - Architecture Entity Extractor       │
                      │  - Inconsistency Detection Rules       │
                      └────┬─────────────┬────────────────┬────┘
                           │             │                │
            ┌──────────────▼───┐  ┌──────▼──────┐  ┌──────▼─────────────┐
            │   PyMuPDF (fitz) │  │  ChromaDB   │  │   SQLite Database  │
            │  - Text blocks   │  │  (Vectors)  │  │   - Projects/Docs  │
            │  - Heading rules │  │  - Isolated │  │   - Entities/Ports │
            │  - Table parser  │  │    by Doc   │  │   - Audit Trail    │
            └──────────────────┘  └─────────────┘  └────────────────────┘
                                         ▲
                                         │ Local Inference
                      ┌──────────────────┴─────────────────────┐
                      │    Local Ollama Server (Port 11434)    │
                      │   Model: llama3.2:3b (configurable)    │
                      │  Embeddings: BAAI/bge-small-en-v1.5    │
                      └────────────────────────────────────────┘
```

---

## 🔑 Core Features

1. **Section-Aware Chunking & Table Extraction**
   - Preserves `Document -> Page -> Section -> Chunk` structural hierarchy.
   - Preserves table boundaries and formats tables as structured markdown chunks with metadata `source_type: "table"`.
2. **Strict Grounding & Zero Hallucination Principle**
   - Enforces engineering-assistant boundaries: never invents components, interfaces, or signals.
   - If evidence is not in the uploaded HLD, returns:
     `INSUFFICIENT EVIDENCE: The uploaded HLD does not provide enough evidence to answer this question.`
3. **Exact Page & Section Citations**
   - Every response provides verifiable source markers: `[Page 21 | Section: Communication Architecture]`.
   - Expandable retrieved evidence viewer displays exact similarity scores and raw chunks.
4. **Explainable Confidence Indicator**
   - Derived mathematically from cosine similarity and multi-chunk support (`High`, `Medium`, `Low`, `Insufficient`).
5. **Architecture Entity Inventory**
   - Automatic extraction and cataloging of:
     - **Components:** Application SWCs, Service SWCs, Sensor/Actuator SWCs.
     - **Interfaces:** Sender-Receiver (S/R) and Client-Server (C/S).
     - **Ports:** P-Ports (Provide) and R-Ports (Require).
     - **Signals:** Data elements, data types, min/max specifications.
     - **Dependencies:** Functional flows and component-to-component mappings.
6. **Cross-Section Potential Inconsistency Detection**
   - Automatically cross-references entities across sections to detect:
     - Conflicting port directions (e.g. P-Port vs R-Port across sections).
     - Interface mismatches (same component/port bound to different interfaces).
     - Undeclared dependency targets.
     - Conflicting signal data types.
7. **Document Isolation & Version Control**
   - SHA-256 fingerprinting prevents redundant re-indexing and tracks revisions.
   - Isolated ChromaDB vector retrieval ensures Document A never accesses Document B.
8. **Multi-Format Export & Immutable Audit Log**
   - Download inventory as CSV or multi-sheet Excel (`.xlsx`).
   - SQLite event log tracks all document uploads, indexing cycles, and engineering queries.

---

## 📋 Prerequisites

- **Operating System:** Windows 10/11 (or Linux/macOS)
- **Python:** Python 3.10, 3.11, or 3.12 (Python 3.11 recommended)
- **Ollama:** (Optional for neural phrasing; fallback extraction operates even without Ollama running)

---

## ⚙️ Installation & Setup (Windows PowerShell)

### Step 1: Open PowerShell and Navigate to Project Directory
```powershell
cd c:\TCS_TATA
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
.\.venv\Scripts\pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env` (already configured by default):
```powershell
Copy-Item .env.example .env
```

### Step 5: (Optional) Setup Ollama for Local Neural LLM
1. Download Ollama from [https://ollama.ai/download](https://ollama.ai/download) and install.
2. In PowerShell, pull the default lightweight model:
```powershell
ollama pull llama3.2:3b
```
3. Start the Ollama daemon:
```powershell
ollama serve
```
*(Note: If Ollama is not running, the application gracefully activates its local deterministic extraction engine, ensuring zero disruptions during testing).*

---

## 🚀 Running the Application

### Option A: One-Click Startup (PowerShell)
To launch both the FastAPI backend and Streamlit UI in separate windows:
```powershell
.\run_all.ps1
```

### Option B: Manual Startup

**Terminal 1 — Start FastAPI Backend:**
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Swagger API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

**Terminal 2 — Start Streamlit Frontend:**
```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app/app.py --server.port 8501
```
- Web Application: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Testing and Verification

Run the comprehensive automated test suite covering PDF parsing, section chunking, vector indexing, document isolation, grounding, citations, and API endpoints:

```powershell
.\.venv\Scripts\pytest -v
```

---

## 🖥️ User Guide & Workflow Walkthrough

### 1. Ingesting an HLD Document
1. Navigate to **Upload HLD** in the sidebar.
2. Either upload an existing `.pdf` file or click **⚡ Generate & Ingest Synthetic Sample**.
3. Observe real-time indexing metrics:
   ```
   ✓ Document indexed
   Pages: 4
   Sections detected: 7
   Chunks created: 12
   Entities detected: 28
   Status: Indexed ✓
   ```

### 2. Asking Architectural Questions
1. Navigate to **Ask the HLD**.
2. Select or type an engineering query:
   - *"Which software components communicate with the CAN interface?"*
   - *"What signals are transmitted over the BatteryStatusIf interface?"*
   - *"What are the P-Ports and R-Ports defined for VehicleControlSWC?"*
   - *"What is the cryptographic satellite uplink protocol?"* (Notice: returns `INSUFFICIENT EVIDENCE`)
3. View the grounded answer, confidence badge, source citations (`[Page X | Section: Y]`), and expandable raw chunk evidence.

### 3. Reviewing Architecture Inventory
1. Navigate to **Architecture Inventory**.
2. Explore tabs: **Components**, **Interfaces**, **Ports**, **Signals**, and **Dependencies**.
3. Click **📥 Download CSV** or **📊 Download Excel** to export full spreadsheets for downstream engineering workflows.

### 4. Inspecting Potential Inconsistencies
1. Navigate to **Analysis & Inconsistencies**.
2. Inspect detected discrepancies (e.g., conflicting port directions across sections).
3. Review Evidence 1 vs Evidence 2 with section links.

### 5. Reviewing Audit Trail
1. Navigate to **Audit Trail**.
2. Inspect the immutable audit log of document uploads, queries, confidence scores, and export events.

---

## 🔒 Security & Privacy Guarantees

- **100% Local Inference:** No document text or embeddings are sent to external cloud APIs (e.g. OpenAI or Gemini).
- **Document Isolation:** ChromaDB queries strictly filter by `where={"document_id": str(doc_id)}`. Cross-document leakage is mathematically prohibited.
- **Sanitized Filenames & Inputs:** Path traversal vulnerabilities are mitigated via filename sanitization.
- **Fail-Safe Logging:** Full document texts and raw confidential code blocks are never printed to public logs.

---

## 👥 Presentation Demonstration Plan (5–10 Minutes)

1. **System Overview (1 min):** Highlight the challenge of navigating 100+ page AUTOSAR HLD documents and introduce the architecture.
2. **Ingestion & Section Chunking (2 min):** Click "Generate & Ingest Synthetic Sample" or upload a PDF. Show SHA-256 fingerprinting, section detection, and table extraction.
3. **Grounded Q&A & Citations (3 min):**
   - Ask: *"Which components communicate with CAN?"* -> Show answer and `[Page 2 | Section: Communication Architecture]` citations.
   - Ask an unsupported question: *"What is the satellite uplink key?"* -> Highlight strict adherence to `INSUFFICIENT EVIDENCE`.
4. **Architecture Inventory & Multi-Sheet Excel Export (2 min):** Show extracted SWCs, ports, signals, and download `.xlsx`.
5. **Inconsistency Detection & Audit Log (2 min):** Show how the system caught the conflicting port direction in Section 7 and inspect the SQLite audit trail.
