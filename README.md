# RAG Pipeline for AI Governance Documents

A retrieval-augmented generation (RAG) pipeline that answers questions about long policy documents and shows which passages each answer came from. It was built in Python with LangChain, FAISS, and Azure OpenAI.

Built as a course-end project for the **Post Graduate Program in Applied Agentic AI** (Virginia Tech and Simplilearn).

## The problem

Policy reports on AI run to dozens of pages each, and a general-purpose LLM hasn't read them, especially recent ones. This project grounds an LLM's answers in a specific document set: it retrieves the most relevant passages at question time and returns them alongside the answer, so the answer can be checked against its sources.

## Documents

Three public reports, 159 pages in total:

- **NIST AI Risk Management Framework (AI RMF 1.0)**, 48 pages
- **Preliminary Report of the Independent International Scientific Panel on AI** (UN, July 2026), 59 pages
- **UNDP Digital Strategy 2022-2025**, 52 pages

The PDFs are not included in this repo. See [`data/README.md`](data/README.md) for download links.

## How it works

```
PDFs ──► PyPDFLoader ──► RecursiveCharacterTextSplitter ──► Azure OpenAI embeddings ──► FAISS index
 (one document per page)   (2,000-char chunks, 200 overlap)                                   │
                                                                                               ▼
                                    Answer + source chunks ◄── gpt-5-mini ◄── top 4 similar chunks ◄── question
```

1. **Load.** Each PDF page becomes a LangChain document, keeping the file name and page number.
2. **Chunk.** Pages are split into chunks of up to 2,000 characters with 200 characters of overlap, breaking at paragraphs or lines where possible.
3. **Embed and store.** Chunks are embedded with Azure OpenAI and stored in an in-memory FAISS index.
4. **Retrieve and answer.** A LangChain `RetrievalQA` chain finds the 4 most similar chunks, passes them to the chat model, and returns both the answer and the chunks it used.

## Results

| Metric | Value |
|---|---|
| Pages loaded | 159 |
| Chunks created | 309 |
| Average chunk size | about 1,462 characters |
| Chunks retrieved per query | 4 |

**Sample query:** *"What are the primary findings of the reports on AI?"*

The model gave a fluent, well-organized answer covering capabilities, concentration of frontier AI development, economic effects, risks, and evidence gaps. However, **all 4 retrieved chunks came from the UN report**, so the NIST and UNDP documents were not represented in the answer. Plain top-k similarity search returns the closest chunks regardless of which document they're from. A fluent answer is not evidence of balanced retrieval, and checking the returned sources is what exposed the gap.

The full write-up, including per-document metrics, is in [`report/report.md`](report/report.md).

## Challenges

- **Rate limits.** Sending all 309 chunks to the embedding endpoint at once failed with rate-limit errors. Embedding in batches of 5 with a 15-second pause between batches worked, taking about 15 minutes.
- **Separating auth errors from rate limits.** Intermittent `401` errors made it unclear whether the key or the request volume was the problem. A 2-chunk test call showed the key was valid, which pointed to rate limiting.
- **Provider change.** The brief specified the OpenAI API, but the working credentials were for Azure OpenAI, so the pipeline uses LangChain's Azure classes. The RAG logic is the same either way.

## What I'd improve next

- Retrieve a balanced set of chunks across documents, or let users filter by document
- Add hybrid keyword + vector search (e.g., BM25) and a re-ranking step
- Build a small evaluation set of questions with known answer locations and measure retrieval accuracy
- Tune chunk size and overlap against that evaluation set
- Include file and page citations in every answer

## Run it yourself

Requires Python 3.12 and access to an Azure OpenAI resource with a chat deployment and an embedding deployment.

```bash
git clone https://github.com/<your-username>/rag-pdf-pipeline.git
cd rag-pdf-pipeline
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your own credentials to .env
```

Download the three PDFs into `data/` (see [`data/README.md`](data/README.md)), then open `rag_pipeline.ipynb` and run the cells in order. Embedding takes about 15 minutes because of the batching.

The saved outputs in the notebook are from the original run. The notebook was originally run in VS Code against an Azure OpenAI gateway provided by the course, which is no longer used here; you'll need your own credentials.

## Tech stack

Python · LangChain · FAISS · Azure OpenAI (gpt-5-mini, embeddings) · pypdf · Jupyter
