# rag_utils.py
import os
import math
from uuid import uuid4
from datetime import datetime
from openai import OpenAI
import chromadb

# --- Configuration ---
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000        # chars per chunk (tweakable)
CHUNK_OVERLAP = 200      # chars overlap
PERSIST_DIR = "chroma_store"  # persistent DB folder

# --- Clients ---
openai_client = OpenAI()
# Use PersistentClient (simpler API). If not available, use chromadb.Client with Settings.
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)


# --- Helpers: chunking ---
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


# --- Build or update a collection for one episode ---
def upsert_episode_to_vector_store(transcript_path, episode_id=None, title=None, source_url=None):
    """
    Creates (or updates) a collection for this episode and inserts chunk embeddings.
    episode_id: unique id for the episode (if None, we generate one).
    title, source_url: stored as metadata for each chunk.
    Returns the chroma collection object.
    """
    metadata = {
        "episode_id": episode_id or str(uuid4()),
        "title": title or os.path.splitext(os.path.basename(transcript_path))[0],
        "source_url": source_url or ""
    }

    # ensure collection name deterministic per episode
    coll_name = f"episode_{metadata['episode_id']}"
    collection = chroma_client.get_or_create_collection(name=coll_name)

    # load transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"Building embeddings for {len(chunks)} chunks...")

    ids = []
    docs = []
    embs = []
    metas = []

    # Create embeddings in batches to be efficient
    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        # OpenAI embeddings call (returns data list)
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
        for j, emb_obj in enumerate(resp.data):
            emb = emb_obj.embedding
            chunk_id = str(uuid4())
            ids.append(chunk_id)
            docs.append(batch[j])
            embs.append(emb)
            metas.append({
                "episode_id": metadata["episode_id"],
                "title": metadata["title"],
                "source_url": metadata["source_url"],
                "created_at": datetime.utcnow().isoformat()
            })

    # add to collection (chroma)
    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    print(f"✅ Upserted {len(ids)} chunks to collection: {coll_name}")
    return collection


# --- Query / Retrieval ---
def retrieve_relevant_chunks(query, collection, top_k=4):
    """Return top_k documents (strings) from the collection given a query."""
    q_emb = openai_client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents","metadatas","distances"])
    # chroma query returns structured results: res["documents"][0] is list
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[]])[0]
    return list(zip(docs, metas, dists))


# --- RAG Chat: retrieve + generate ---
def rag_answer(question, collection, top_k=4, model="gpt-4o-mini"):
    """
    Retrieves relevant chunks and prompts the LLM to answer based on them.
    Returns the assistant answer string and the retrievals used.
    """
    retrieved = retrieve_relevant_chunks(question, collection, top_k=top_k)
    context_text = "\n\n---\n\n".join([f"[chunk]\n{doc}" for doc, meta, dist in retrieved])

    prompt = f"""
You are a helpful assistant answering questions about a podcast episode using only the provided transcript excerpts.
If the answer cannot be found in the excerpts, say "I don't know from the transcript" and offer to summarize or fetch more context.

Transcript excerpts:
{context_text}

Question:
{question}
"""
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise assistant that cites the transcript when possible."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    answer = response.choices[0].message.content.strip()
    return answer, retrieved


# --- Utilities for multi-episode store ---
def get_collection_for_episode(episode_id):
    name = f"episode_{episode_id}"
    try:
        coll = chroma_client.get_collection(name=name)
        return coll
    except Exception:
        return None


def list_collections():
    # returns list of collection names (may vary by chroma version)
    try:
        return chroma_client.list_collections()
    except Exception:
        # fallback: try to access internal collections property
        return [c.name for c in chroma_client.get_collections()]

def upsert_to_global_index(transcript_path, title, source_url, episode_id=None):
    """
    Add an episode's transcript to the global podcast knowledge base.
    Each chunk stores metadata: episode_id, title, source_url.
    """
    GLOBAL_COLLECTION_NAME = "podcast_knowledge_base"
    EMBED_MODEL = "text-embedding-3-small"

    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(GLOBAL_COLLECTION_NAME)
    openai_client = OpenAI()

    # Read transcript and chunk
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)

    print(f"Indexing {len(chunks)} chunks into global collection...")

    batch_size = 16
    ids, docs, embs, metas = [], [], [], []

    episode_id = episode_id or str(uuid4())

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
        for j, emb_obj in enumerate(resp.data):
            ids.append(str(uuid4()))
            docs.append(batch[j])
            embs.append(emb_obj.embedding)
            metas.append({
                "episode_id": episode_id,
                "title": title,
                "source_url": source_url
            })

    collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    print(f"✅ Added {len(ids)} chunks from '{title}' to global index.")
    return collection

def query_global_index(query, top_k=6, model="gpt-4o-mini"):
    """
    Query the global knowledge base to answer cross-podcast questions.
    Retrieves top_k relevant chunks from all episodes, and synthesizes an answer.
    """
    GLOBAL_COLLECTION_NAME = "podcast_knowledge_base"
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    openai_client = OpenAI()
    collection = chroma_client.get_collection(GLOBAL_COLLECTION_NAME)

    q_emb = openai_client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas"])

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n---\n\n".join(
        [f"[{m['title']}] {d}" for d, m in zip(docs, metas)]
    )

    prompt = f"""
You are an expert research assistant synthesizing ideas from multiple podcasts.

Use the following transcript excerpts (from different episodes) to answer the user's question.
Be specific and cite episode titles in your response.

Context:
{context}

Question:
{query}

Write your answer clearly, and if the context is insufficient, say so.
"""

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an analytical podcast summarizer and synthesizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content.strip()
    return answer, list(zip(docs, metas))
