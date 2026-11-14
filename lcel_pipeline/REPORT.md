# LCEL Pipeline: LangChain Expression Language & Document Chunking Architecture

---

## 1. Problem Statement & Project Details
While raw SDK integrations (built in Project 1) provide total control over network sockets, retries, and telemetry, production AI applications frequently standardize on orchestrators like **LangChain** and **LangChain Expression Language (LCEL)** to achieve component modularity, declarative composition, and unified interfaces across heterogeneous providers.

However, frameworks introduce their own architectural risks: hidden failure points, tight coupling, and rapid API churn. **LCEL Pipeline** is an engineering exploration of LangChain's core primitives:
1. **LCEL Runnables & Chains**: Decoupling prompts from models using `ChatPromptTemplate | ChatGroq | StrOutputParser` to enable model-agnostic parameterization and zero-API prompt inspection (`--show-prompt`).
2. **Document Ingestion & Text Splitting**: Ingesting text files with `TextLoader` and decomposing documents with `RecursiveCharacterTextSplitter(add_start_index=True)` to observe chunk fragmentation, context starvation, and context dilution.
3. **Strict Error Boundaries**: Enforcing exception handling exclusively at the CLI boundary, ensuring chains remain pure and reusable across CLI, web, and background worker contexts.
4. **Empirical Boundary Analysis**: Benchmarking fine-grained (400/50) vs coarse-grained (1200/200) chunking on a 1,343-word technical corpus (`notes.txt`) across 5 RAG architecture topics.

---

## 2. Tech Stack, Tools & Libraries
- **Language**: Python 3.12+
- **Framework & Primitives**: `langchain-core==1.8.0` (`Runnable`, `ChatPromptTemplate`, `StrOutputParser`, `Document`)
- **LLM Integration**: `langchain-groq==1.1.3` (`ChatGroq`)
- **Text Splitters**: `langchain-text-splitters==1.1.3` (`RecursiveCharacterTextSplitter`)
- **Document Loaders**: `langchain-community==0.4.2` (`TextLoader`)
- **Configuration & Validation**: `pydantic` & `pydantic-settings`
- **CLI Argument Parsing**: `argparse` with inherited parent parsers for global flag reuse
- **Testing**: `pytest`, `pytest-asyncio`, `pytest-mock` (55 passing tests across repo)
- **Dependency Management**: `uv` with pinned `requirements.txt`

---

## 3. Architecture Details

### Architecture Flowchart
```
                           +----------------------------------------+
                           |  CLI Subcommands: explain | document   |
                           +-------------------+--------------------+
                                               |
                                               v
                               +-------------------------------+
                               |  PipelineSettings (config.py) |
                               |   - GROQ_API_KEY validation   |
                               |   - Overlap <= Size invariant |
                               +---------------+---------------+
                                               |
                       +-----------------------+-----------------------+
                       |                                               |
                       v                                               v
          [Subcommand: explain]                              [Subcommand: document]
                       |                                               |
                       v                                               v
+-----------------------------------------------+   +-----------------------------------------------+
| ChatPromptTemplate (topic, audience, format)  |   | TextLoader(encoding="utf-8")                  |
|   Input:  {"topic", "audience", "format"}     |   |   Reads target .txt into Document             |
|   Output: ChatPromptValue (Messages)          |   |   metadata: {"source": filepath}              |
+----------------------+------------------------+   +----------------------+------------------------+
                       |                                               |
                       v                                               v
+-----------------------------------------------+   +-----------------------------------------------+
| ChatGroq(model, temperature=0.0)              |   | RecursiveCharacterTextSplitter                |
|   Input:  ChatPromptValue                     |   |   chunk_size, chunk_overlap                   |
|   Output: AIMessage                           |   |   add_start_index=True                        |
+----------------------+------------------------+   +----------------------+------------------------+
                       |                                               |
                       v                                               v
+-----------------------------------------------+   +-----------------------------------------------+
| StrOutputParser()                             |   | SplitSummary & Formatting                     |
|   Input:  AIMessage                           |   |   Outputs total chunks, character lengths,    |
|   Output: str                                 |   |   start_index offsets, and preview blocks     |
+----------------------+------------------------+   +-----------------------------------------------+
                       |
                       v
+-----------------------------------------------+
| CLI Error Boundary (errors.py)                |
|   Catches Auth (401), RateLimit (429), Network|
|   Outputs short terminal message, exits code 1|
+-----------------------------------------------+
```

### LCEL Stage Contracts
| Stage | Component | Input Type | Output Type | Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | `ChatPromptTemplate` | `dict[str, str]` | `ChatPromptValue` | Validates required keys (`topic`, `audience`, `format`), interpolates strings, and builds typed `SystemMessage` and `HumanMessage` objects. |
| **Stage 2** | `ChatGroq` | `ChatPromptValue` | `AIMessage` | Serializes message list to provider JSON payload, sets `temperature=0`, and issues API call over HTTP. |
| **Stage 3** | `StrOutputParser` | `AIMessage` | `str` | Extracts string from `.content`, discarding token metadata and headers for downstream consumption. |

---

## 4. Key Terminology & Definitions
- **Runnable**: The unified interface in LangChain implemented by prompts, models, parsers, and chains. Exposes `.invoke()`, `.stream()`, `.batch()`, and async equivalents (`.ainvoke()`, `.astream()`, `.abatch()`).
- **LCEL Pipe (`|`)**: A syntactic overload of Python's binary OR operator (`__or__` / `__ror__`) that sequences two `Runnable` objects, passing the output of stage $N$ directly as input to stage $N+1$.
- **ChatPromptTemplate**: A structured message template that produces conversation lists (`BaseMessage`) rather than raw text strings, preserving system/user/assistant boundaries.
- **StrOutputParser**: A lightweight transformation runnable that unwraps an `AIMessage` and returns its text content.
- **Document**: The fundamental unit of text representation in LangChain consisting of `page_content: str` and `metadata: dict`.
- **Recursive Character Splitting**: A splitting algorithm that attempts delimiters in order of semantic preservation (`\n\n` $\rightarrow$ `\n` $\rightarrow$ ` ` $\rightarrow$ `""`), only fragmenting sentences or words as an absolute last resort.
- **Chunk Size vs. Overlap**: `chunk_size` determines maximum character length per chunk; `chunk_overlap` replicates the trailing characters of chunk $N$ into the start of chunk $N+1$ to preserve concepts straddling borders.
- **Context Starvation vs. Context Dilution**:
  - *Starvation*: Chunks are too small (e.g. 200 chars), stripping critical context, antecedents, and conditions.
  - *Dilution*: Chunks are too large (e.g. 2,000 chars), diluting the embedding with irrelevant text and reducing cosine similarity.
- **`start_index`**: The exact character offset of a chunk relative to the source document, critical for citations and UI highlighting.

---

## 5. Section-by-Section Plain-Language Walkthrough

### Section 1: Skeleton and Settings (`config.py` & `cli.py`)
- Configured a two-mode CLI supporting `explain` and `document` subcommands.
- Built a shared options parent parser so global settings (`--model`, `--chunk-size`, `--chunk-overlap`) can be placed anywhere on the command line.
- Enforced mathematical invariant: `chunk_overlap <= chunk_size`. Overlaps larger than size exit with a friendly error and code 1 without a stack trace.
- Pinned `requirements.txt` using `uv pip freeze`.

### Section 2: Explain Mode (`explain.py`)
- Created `ChatPromptTemplate.from_messages` with `{topic}`, `{audience}`, and `{format}`.
- Composed the chain: `prompt | ChatGroq(temperature=0) | StrOutputParser()`.
- Added `--show-prompt` flag to render formatted prompt messages without initiating network calls or burning API credits.
- Demonstrated that changing only the `audience` variable dynamically shifts the explanation from a 5-year-old ("friendly guard") to software engineers ("token-bucket / leaky-bucket") to corporate executives ("traffic-control mechanism").

### Section 3: Document Mode (`document.py`)
- Loaded `.txt` files using `TextLoader` with explicit UTF-8 encoding.
- Decomposed documents into chunks using `RecursiveCharacterTextSplitter(add_start_index=True)`.
- Printed structured reports: character count, total chunks, preview chunks, lengths, metadata, and start indices.
- Handled edge cases gracefully: missing files and empty/whitespace documents print clean errors to `stderr` with code 1.

### Section 4: Error Boundaries (`errors.py`)
- Mapped upstream provider exceptions to distinct, user-friendly messages:
  - **Auth**: `Error: Authentication failed. Please check that your GROQ_API_KEY is valid.`
  - **Rate Limit**: `Error: Rate limit exceeded. Please wait a moment before trying again.`
  - **Network/Timeout**: `Error: Network connection failed. Please check your internet connection.`
- Exception handling lives strictly at the CLI boundary (`cli.py`), keeping `explain.py` pure and composable.
- Automated tests verify that mock exceptions trigger their specific branches without printing tracebacks.

### Section 5: Notes & Comparative Experiment (`notes.txt` & `experiment.py`)
- Authored `notes.txt` (1,343 words across 5 technical topics: Vector Metric Spaces, HNSW, Chunking Dynamics, Hybrid RRF, Cross-Encoders).
- Compared two chunking settings:
  - **400 / 50**: Produced **37 chunks** (avg 287 chars).
  - **1200 / 200**: Produced **14 chunks** (avg 717 chars).
- Identified concrete technical concepts severed across boundaries in 400/50:
  - `SIMD inner product matrix multiplications (such as AVX-512 or ARM Neon instructions)` (severed across Chunk #5 and #6).
  - `Distributional Hypothesis & InfoNCE` (severed across Chunk #2 and #3).
  - `HNSW Graph Traversal & M/efConstruction` (severed across Chunk #12 and #14).
- In 1200/200, smaller technical concepts remained intact inside single chunks, but large chunks (#3, #6, #9) straddled top-level topic boundaries.

---

## 6. Consolidated Failure Modes & Defenses

| Failure Mode | Root Cause | Defense Implemented in Project |
| :--- | :--- | :--- |
| **Overlap > Size** | User enters `--chunk-overlap 500 --chunk-size 200`. | `validate_chunk_settings()` catches invariant upfront, prints friendly error, exits code 1 without stack trace. |
| **Missing API Key** | `GROQ_API_KEY` is empty or missing. | `load_settings(require_api_key=True)` validates string, exits code 1 with clean message. |
| **Invalid Auth (401)** | Expired or incorrect Groq token. | CLI error boundary catches `AuthenticationError`, displays `AUTH_ERROR_MSG`. |
| **Rate Limit (429)** | RPM / TPM request quotas exceeded. | CLI error boundary catches `RateLimitError`, displays `RATE_LIMIT_ERROR_MSG`. |
| **Network Disconnect** | Dropped TCP socket or DNS failure. | CLI error boundary catches `APIConnectionError` / `TimeoutException`, displays `NETWORK_ERROR_MSG`. |
| **Missing Document** | User supplies path to non-existent file. | `load_and_split_document` checks `os.path.exists`, prints clean `FileNotFoundError` message. |
| **Empty File** | File is 0 bytes or whitespace-only. | Pre-split validation checks `docs[0].page_content.strip()`, exits with friendly message. |
| **Template Placeholder Mismatch** | Dict passed to chain lacks required keys. | `run_explain` validates and maps inputs; `test_build_explain_prompt_placeholders` guarantees input contract. |
| **Unescaped Braces in Format** | Prompt format spec contains JSON curly brackets (`{}`). | Prompts use clean separation between system instructions and format parameters. |
| **Non-UTF-8 Encoding** | Text file has Windows-1252 or Latin-1 characters. | `TextLoader(cleaned_path, encoding="utf-8")` guarantees consistent UTF-8 decoding. |

---

## 7. How to Run, Use, and Verify

### Prerequisites & Setup
```bash
# Clone & enter directory
git clone git@github.com:sharmavaibhav31/llm-engineering-hub.git
cd llm-engineering-hub

# Virtual environment & dependencies
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 1. Explain Mode
```bash
# A. Inspect formatted prompt without calling the API
python -m lcel_pipeline explain "Rate Limiting" -a "senior engineers" -f "bullet points" --show-prompt

# B. Run with target audience: 5-year-old
python -m lcel_pipeline explain "Rate Limiting" -a "5-year-old" -f "1 sentence"

# C. Run with target audience: software engineers
python -m lcel_pipeline explain "Rate Limiting" -a "software engineers" -f "1 sentence"

# D. Run with target audience: corporate executive
python -m lcel_pipeline explain "Rate Limiting" -a "corporate executive" -f "1 sentence"
```

### 2. Document Mode
```bash
# Ingest and chunk a document with custom chunk size and overlap
python -m lcel_pipeline document lcel_pipeline/notes.txt --chunk-size 400 --chunk-overlap 50 -n 3
```

### 3. Chunking Experiment
```bash
# Run comparative 400/50 vs 1200/200 experiment
PYTHONPATH=. python lcel_pipeline/experiment.py
```

### 4. Running Test Suite
```bash
# Run all 55 tests across the repository
pytest
```

---

## 8. Defendable Points (Interview Preparation)

1. **What does the pipe operator (`|`) do in LCEL?**
   - *Defense*: The pipe operator chains two `Runnable` objects into a `RunnableSequence`. It passes the output of the left runnable as the input to the right runnable. Because the resulting sequence is itself a `Runnable`, it automatically inherits `.invoke()`, `.stream()`, `.batch()`, and async versions without needing manual event loop or streaming orchestration.
2. **Why separate the prompt template from the model?**
   - *Defense*: Hardcoding prompts inside model calls couples your prompt logic to a specific provider. Decoupling prompts via `ChatPromptTemplate` treats prompts as testable data. We can inspect and unit test prompt rendering offline (`--show-prompt`), swap underlying providers (Groq, Anthropic, local vLLM) in one line, and version-control prompts independently.
3. **How do chunk size and overlap affect vector retrieval quality?**
   - *Defense*: Small chunks (200–400 chars) have high semantic density and retrieve exact factual answers, but risk context starvation. Large chunks (1200–2000 chars) provide rich narrative context, but suffer from context dilution where disparate topics blur the embedding. Chunk overlap acts as an insurance policy, ensuring concepts that fall across cut lines appear whole in at least one adjacent chunk.
4. **Why `RecursiveCharacterTextSplitter` over fixed-size chunking?**
   - *Defense*: Fixed-size chunking blindly slices text at character index $N$, frequently severing words, code blocks, or sentences in half. The recursive splitter tries a priority cascade of natural boundaries (`\n\n`, `\n`, ` `), preserving paragraph and sentence integrity before resorting to character cuts.
5. **Where should exception handling live in framework pipelines?**
   - *Defense*: Exception handling must live at the outer application/CLI boundary, never inside reusable chain definitions. If `explain.py` caught exceptions and printed to stdout, it could never be safely embedded in a FastAPI route, Celery worker, or batch job. The core pipeline should fail fast and let outer boundaries decide whether to retry, return HTTP error codes, or render terminal alerts.
6. **When would you NOT use LangChain?**
   - *Defense*: For simple single-call completions or highly custom socket-level streaming and retry requirements (as built in Project 1), the raw SDK is lighter, faster to initialize, and has zero dependency bloat or breaking API shifts. LangChain is justified when composing multi-stage pipelines (RAG, memory, tool routing) where standardizing on the `Runnable` protocol saves weeks of custom boilerplate.

---

## 9. Empirical Resume Bullets

> *Measured and verified against the live Groq API and test suite:*

- **Built a multi-mode CLI on LangChain (LCEL) composing ChatPromptTemplate, ChatGroq and StrOutputParser, parameterized by topic, audience and format with zero-API prompt inspection.**
- **Implemented document ingestion with TextLoader and RecursiveCharacterTextSplitter and compared 2 chunk-size and overlap settings (400/50 yielding 37 chunks vs 1200/200 yielding 14 chunks) on boundary quality, identifying 3 severed technical concepts.**
