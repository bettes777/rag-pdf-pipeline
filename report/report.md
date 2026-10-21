# Building a RAG Pipeline Using PDF Chunking and Retrieval

**Course:** Applied Agentic AI  |  **Project:** Course-End Project
**Author:** Betty

---

## 1. Introduction

**Objective.** The goal of this project is to build a Retrieval-Augmented Generation (RAG) pipeline that lets a user ask a question about a set of long PDF documents and get back the most relevant passages. The pipeline loads PDFs, splits them into chunks, converts the chunks into vector embeddings, stores them in a FAISS vector database, and retrieves the closest chunks to a query using semantic search.

**Relevance of RAG.** Large language models only know what was in their training data, and they can produce confident but unsupported answers. RAG addresses this by retrieving relevant text from a trusted document collection at question time and supplying it to the model as context. This grounds answers in specific sources, lets the system work with documents the model never saw in training, and makes answers traceable to a file and page. Long documents cannot be passed to a model in full, so chunking and retrieval are what make the approach practical.

**Documents used.** Three publicly available documents on AI governance and digital development were used:

| Document | Pages | Notes |
|---|---|---|
| NIST AI Risk Management Framework (AI RMF 1.0), NIST AI 100-1 | 48 | January 2023 |
| Preliminary Report of the Independent International Scientific Panel on AI | 59 | Dated July 2026 on its cover; the final page has no extractable text |
| UNDP Digital Strategy | 52 | Cover reads 2022-2025 |
| **Total** | **159** | |

Page counts and the empty final page of the UN report were checked by extracting text with `pypdf`, the library `PyPDFLoader` relies on. All three files contain a text layer, so no OCR was needed.

## 2. Implementation Overview

**Tools.** Visual Studio Code (with a Python virtual environment), LangChain, FAISS (`faiss-cpu`), and Azure OpenAI (both for embeddings and for the chat model), accessed through an Azure API Management gateway endpoint provided for the course.

**Pipeline steps** (all in `rag_pipeline.ipynb`):

1. **Load.** `PyPDFLoader` reads each of the 3 PDFs and returns one `Document` per page, with the source file and page number kept as metadata.
2. **Chunk.** `RecursiveCharacterTextSplitter` splits pages into chunks of up to 2000 characters with 200 characters of overlap. It tries to break on paragraph boundaries first, then lines, then spaces, so chunks tend to end at natural boundaries rather than mid-sentence. The overlap means a sentence cut at a chunk boundary still appears whole in at least one neighboring chunk. This produced 309 chunks in total.
3. **Embed.** Each chunk is embedded with Azure OpenAI's `text-embedding-ada-002` deployment.
4. **Store.** The embeddings and chunk text are stored in a FAISS vector index in memory for the session.
5. **Retrieve.** `RetrievalQA` wraps the FAISS store as a retriever using the default setting, which returned the 4 most similar chunks per query, and passes them to the chat model.
6. **Generate.** An Azure OpenAI chat deployment answers using only the retrieved chunks, and the chain also returns the source documents used.

**Parameter choices.** Chunk size (2000) and overlap (200) were starting values suggested for this exercise, not tuned against a benchmark — a genuine area for improvement noted in Section 4. The number of retrieved chunks was left at the retriever's default (4) and was likewise not tuned.

**A deviation from the original plan worth noting.** The course-end project instructions specify the OpenAI API, and that was the original intent for this pipeline. In practice, the only working credentials available were for an Azure OpenAI deployment reached through an API management gateway, so the final pipeline uses `AzureOpenAIEmbeddings` and `AzureChatOpenAI` rather than the plain OpenAI client. The RAG logic (chunk → embed → store → retrieve → generate) is unaffected; only the embedding/chat provider differs from the brief.

## 3. Metrics and Retrieval Information

**Chunking metrics.**

| Metric | Value |
|---|---|
| Documents loaded | 3 (NIST AI RMF, UN Scientific Panel preliminary report, UNDP Digital Strategy) |
| Chunk size / overlap setting (max characters per chunk) | 2000 / 200 |
| **Total chunks created** | **309** |
| **Average chunk size** | **about 1,462 characters** (median about 1,905; range 45 to 1,999) |

| Document | Pages | Chunks | Average chunk size (characters) |
|---|---|---|---|
| NIST AI RMF | 48 | 82 | about 1,368 |
| UN Scientific Panel preliminary report | 59 | 140 | about 1,588 |
| UNDP Digital Strategy | 52 | 87 | about 1,349 |
| **Total** | **159** | **309** | **about 1,462** |

Note: 2000/200 is the *setting* given to the text splitter (the maximum size allowed per chunk), not the measured average. `RecursiveCharacterTextSplitter` breaks at natural boundaries (paragraphs, then lines, then spaces) whenever possible, so actual chunks are typically smaller than this maximum. Short chunks come from pages with little text, such as covers and section dividers.

How the average was measured: it was not printed during the original run, so the same chunking (same PDFs, same splitter settings) was reproduced separately. The reproduction matched the original run exactly on page counts (48, 59, 52) and total chunks (309). The notebook now includes a cell that prints these metrics directly.

**Retrieval configuration.** `vectorstore.as_retriever()` with default settings — each query returned the 4 most similar chunks by embedding distance.

### Sample query output

**Query 1:** "What are the primary findings of the reports on AI?"

The chain returned a coherent answer (reproduced in condensed form below), but checking `result["source_documents"]` showed that all 4 retrieved chunks came from a single document — the UN Scientific Panel preliminary report — rather than spread across all three PDFs. This is expected behavior for plain top-k similarity search: it returns the closest chunks by embedding distance regardless of source file, so a broad query can pull most or all of its matches from whichever document happens to discuss that topic most extensively.

Condensed answer received:
- AI is a rapidly advancing general-purpose technology with large positive potential, but benefits require complementary investments in data, skills, and organizational change.
- Capabilities are advancing faster than the ability to measure, evaluate, and govern them.
- Frontier model development is concentrated among a small number of actors.
- Benefits and inputs are geographically and linguistically uneven.
- Agentic AI systems represent a governance step change with new oversight needs.
- AI can erode shared reality through misinformation and synthetic content, with implications for human rights.

**Assessment:** The answer was fluent and consistent with the source material, but it under-represents the NIST AI RMF and UNDP Digital Strategy documents, since retrieval was not document-balanced. The top-ranked chunk was the UN report's "4. Gaps and next steps" section (printed page 45), followed by "3.3 Economic implications", the table of contents, and "4.3 Next steps". This is a genuine limitation of naive top-k similarity search, not a code defect.

## 4. Conclusion and Lessons Learned

### Insights gained
- The retrieval step is sensitive to the number of chunks retrieved and to query framing: a single broad question tends to concentrate results in whichever document discusses that topic most, while narrower, document-specific questions are more likely to surface the intended source.
- The generated answer's fluency does not by itself confirm balanced retrieval — checking `source_documents` was necessary to see that only one document was actually represented in the context sent to the model.
- Getting the pipeline to run at all required treating infrastructure (API authentication and rate limits) as a first-class part of the exercise, separate from the RAG logic itself — the chunking, embedding, and retrieval code worked correctly throughout; the obstacles were entirely in reliably calling the embedding API.

### Challenges faced
- **Rate limiting on the embedding endpoint.** Sending all 309 chunks to the Azure embedding deployment in one call (or in batches of 250) repeatedly failed or hung due to `RateLimitError`. This was resolved by embedding in small batches (5 chunks at a time) with a 15-second pause between batches, which completed successfully after roughly 15 minutes.
- **Authentication errors on the same endpoint.** A separate `401 AuthenticationError` ("invalid subscription key or wrong API endpoint") appeared intermittently on the same credentials that later succeeded, which made it initially unclear whether the problem was the key itself or the request volume. Testing with a very small batch (2 chunks) first isolated this: the key was valid, and the rate limiting was the actual cause of the earlier failures.
- **Provider deviation from the assignment brief.** The instructions specify the OpenAI API; only Azure OpenAI credentials were available and working, so the pipeline uses `AzureOpenAIEmbeddings` / `AzureChatOpenAI` instead.

### Possible improvements

- **Tune chunking.** Test other chunk sizes and overlaps, or split on document headings so each chunk maps to a section.
- **Clean the text.** Strip repeated headers, footers, and page numbers before chunking.
- **Better retrieval.** Combine keyword search (e.g., BM25) with vector search, and add a re-ranking step on the top results.
- **Metadata filtering.** Let users restrict a query to one document or a page range.
- **Evaluate systematically.** Write a small set of questions with known answer locations and measure how often the correct chunk appears in the top-k.
- **Cite sources.** Return file and page references with every generated answer.
- **Handle tables and figures.** Use a table-aware extractor, since plain text extraction can flatten tables.
