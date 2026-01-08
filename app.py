import streamlit as st
from openai import OpenAI
import chromadb
import os
from dotenv import load_dotenv
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# Load models and clients
client = OpenAI()
chroma = chromadb.PersistentClient(path="chroma_store")
collection = chroma.get_or_create_collection("podcast_index")

st.title("🎧 Podcast Oracle")

query = st.text_input("Ask something about your podcasts:")

if st.button("Ask") and query:
    results = collection.query(query_texts=[query], n_results=3)
    context = "\n\n".join(results["documents"][0])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI trained on Varun’s podcast transcripts."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
    )

    st.write(response.choices[0].message.content)
