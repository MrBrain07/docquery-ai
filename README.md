# DocQuery — RAG-based Q&A Assistant

Upload documents (PDF or TXT), ask questions in plain English, and get answers
grounded in the actual document content — with the exact source passages shown
alongside every answer.

Live demo: **[add your Streamlit Cloud link here after deploying]**

---

## Why this exists

Off-the-shelf LLMs answer from what they memorized during training. They can't
answer questions about a private document set, and they'll confidently make
things up (hallucinate) when they don't know something. **Retrieval-Augmented
Generation (RAG)** fixes this by retrieving the actual relevant passages from
your documents first, then asking the LLM to answer *using only that context*
— and showing you exactly which passages it used, so the answer is verifiable.

## How it works

```
 Upload docs        Ask a question
      |                    |
      v                    v
 [ Chunk text ]     [ Embed question ]
      |                    |
      v                    |
 [ Embed chunks ]          |
      |                    |
      v                    v
 [  FAISS vector index  <-- similarity search  ]
                            |
                            v
                 [ Top-k relevant chunks ]
                            |
                            v
              [ LLM (Groq/Llama) generates answer
                grounded ONLY in those chunks ]
                            |
                            v
                    Answer + cited sources
```

1. **Ingest** — uploaded PDFs/TXT files are loaded and their raw text extracted.
2. **Chunk** — text is split into ~800-character overlapping chunks
   (`RecursiveCharacterTextSplitter`) so retrieval can be precise without
   losing surrounding context.
3. **Embed** — each chunk is converted into a vector using a local
   sentence-transformer model (`all-MiniLM-L6-v2`) — this runs on CPU, no API
   key or cost involved.
4. **Store & retrieve** — vectors are indexed in **FAISS** for fast similarity
   search. At query time, the question is embedded the same way and the top-4
   most similar chunks are retrieved.
5. **Generate** — the question plus retrieved chunks are sent to an LLM
   (GPT-OSS 120B via the **Groq** API, chosen for its free tier and low
   latency) with an explicit instruction to answer only from the provided
   context and to cite which source it used.

## Design decisions worth knowing for an interview

- **Local embeddings, API-based generation**: embedding is cheap and
  privacy-sensitive (your documents never leave your machine at this stage),
  so it runs locally. Generation needs a capable LLM, which is impractical to
  self-host on a free tier, so that step calls an external API.
- **Why Groq specifically**: fast inference and a generous free tier make it
  practical for a demo/portfolio project without a paid OpenAI key.
- **Why show retrieved sources**: transparency. If the retrieval step pulled
  irrelevant chunks, the answer will be visibly wrong or off-topic, and you
  can debug *which* stage failed instead of treating the system as a black box.
- **Known limitations**: no persistent storage (index rebuilds each session,
  fine for a demo, not for production); no re-ranking step after retrieval;
  chunk size is fixed rather than adaptive to document structure. These are
  honest, sensible things to mention if asked "what would you improve?"

## Running locally

```bash
git clone <your-repo-url>
cd docquery
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env      # then paste your free Groq API key into .env
streamlit run app.py
```

Get a free Groq API key at **console.groq.com** (sign up, create an API key —
no credit card needed for the free tier).

> **Note on `requirements.txt`:** versions are pinned exactly, not left open
> with `>=`. LangChain moved `RecursiveCharacterTextSplitter` out of
> `langchain.text_splitter` and into its own `langchain-text-splitters`
> package in a recent major release — an unpinned install pulled that newer
> version and broke the import. Pinning avoids surprise breaks like this when
> Streamlit Cloud rebuilds your environment.


## Deploying it for free (Streamlit Community Cloud)

1. Push this folder to a new **public GitHub repo**.
2. Go to **share.streamlit.io** → sign in with GitHub → "New app".
3. Pick your repo, branch `main`, main file path `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_real_key_here"
   ```
5. Click **Deploy**. In a minute or two you'll get a public URL like
   `https://your-app-name.streamlit.app` — that's your live demo link.
6. Paste that link into your resume's "Live demo" field and into your
   GitHub repo's description.

## Tech stack

Python, Streamlit, LangChain, FAISS, Sentence-Transformers (HuggingFace),
Groq API (GPT-OSS 120B).
