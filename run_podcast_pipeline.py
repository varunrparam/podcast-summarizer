import os
from datetime import datetime
import requests
from dotenv import load_dotenv
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

from podcast_utils import (
    download_podcast,
    transcribe_audio,
    save_transcript,
    summarize_transcript,
    write_summary_to_notion,
)

from rag_utils import upsert_episode_to_vector_store, get_collection_for_episode, rag_answer, upsert_to_global_index, query_global_index

def main():
    # 1️⃣ Get input URL
    youtube_url = input("Enter the YouTube podcast URL: ").strip()
    if not youtube_url:
        print("❌ No URL provided.")
        return

    # 2️⃣ Download podcast audio
    print("\n🎧 Downloading podcast audio...")
    mp3_path = download_podcast(youtube_url)
    print(f"✅ Audio downloaded to: {mp3_path}")

    # 3️⃣ Transcribe audio
    print("\n🗣️ Transcribing audio... (this might take a few minutes)")
    transcript_text = transcribe_audio(mp3_path)

    # 4️⃣ Save transcript
    transcript_file = save_transcript(transcript_text, youtube_url)

    # 5️⃣ Summarize transcript
    print("\n🧠 Generating summary...")
    summary_file, title = summarize_transcript(transcript_file, youtube_url)
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_text = f.read()

    print(f"✅ Summary complete and saved to: {summary_file}")

    # 6️⃣ Write to Notion
    print("\n🪄 Writing summary to Notion...")
    write_summary_to_notion(title, summary_text, link=youtube_url)

    print(f"\n🎉 Done! '{title}' added to your Notion workspace.")

def rag():
    # Suppose you have:
    transcript_path = "transcripts/transcript_20251017_222436.txt"
    episode_id = "1"  # optional
    title = "Content and Community"
    source_url = "https://www.youtube.com/watch?v=vzQ67iakEAA&t=2s"

    # Build/persist vectors (only once per episode, safe to re-run)
    collection = upsert_episode_to_vector_store(transcript_path, episode_id=episode_id, title=title,
                                                source_url=source_url)

    # Later, chat:
    question = input("Enter your question about the podcast: ")
    answer, retrieved = rag_answer(question, collection)
    print("Answer:", answer)
    print("Retrieved chunks (count):", len(retrieved))

def upsertion():
    # upsert_to_global_index(
    #     transcript_path="transcripts/transcript_20251017_222436.txt",
    #     title="Content and Community",
    #     source_url="https://www.youtube.com/watch?v=vzQ67iakEAA&t=2s"
    # )
    # upsert_to_global_index(
    #     transcript_path="transcripts/transcript_20251017_221955.txt",
    #     title="OpenAI's Windows Play",
    #     source_url="https://www.youtube.com/watch?v=-jWGcg5dXQ0&t=39s"
    # )
    upsert_to_global_index(
        transcript_path="transcripts/transcript_20251017_221032.txt",
        title="Is AI Therapy a Horrible Idea?",
        source_url="https://www.youtube.com/watch?v=CO1HJBbNqeM&t=6s"
    )

    answer, context = query_global_index("What are some of the key risks AI poses to society?")
    print(answer)

if __name__ == "__main__":
    upsertion()
