import os
import subprocess
from datetime import datetime, timezone
from openai import OpenAI
import re
from yt_dlp import YoutubeDL
import requests
import json

def download_podcast(url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)

    # Command returns the final output file path directly
    command = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "-o", f"{output_dir}/%(title)s.%(ext)s",
        "--print", "after_move:filepath",
        url
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    file_path = result.stdout.strip()

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Downloaded file not found: {file_path}")

    print(f"🎧 Download complete: {file_path}")
    return file_path


def split_audio(mp3_path, chunk_length=600):  # 600 seconds = 10 minutes
    """Split a long MP3 into chunks using ffmpeg."""
    output_dir = os.path.join(os.path.dirname(mp3_path), "chunks")
    os.makedirs(output_dir, exist_ok=True)

    # Generate command to split audio
    command = [
        "ffmpeg",
        "-i", mp3_path,
        "-f", "segment",
        "-segment_time", str(chunk_length),
        "-c", "copy",
        os.path.join(output_dir, "chunk_%03d.mp3")
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    chunks = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith(".mp3")]
    return chunks


def transcribe_audio(mp3_path):
    """Transcribe a full audio file by splitting into smaller chunks."""
    client = OpenAI()
    chunks = split_audio(mp3_path)
    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"🎧 Transcribing chunk {i + 1}/{len(chunks)}: {chunk}")
        with open(chunk, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file
            )
        full_transcript += transcript.text.strip() + "\n"

    txt_path = mp3_path.replace(".mp3", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_transcript)

    print(f"✅ Full transcript saved to {txt_path}")
    return full_transcript

def save_transcript(text, youtube_url, output_dir="transcripts"):
    """
    Saves transcript text to a timestamped file for later processing.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"transcript_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Source: {youtube_url}\n\n")
        f.write(text)
    print(f"📝 Transcript saved to: {filename}")
    return filename

def get_video_title(url):
    """Fetch the YouTube or podcast title."""
    ydl_opts = {'quiet': True, 'skip_download': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get('title', 'untitled')

def slugify(text):
    """Convert title to a filesystem-safe name."""
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text.lower()

def summarize_transcript(file_path, url, output_dir="summaries"):
    client = OpenAI()

    title = get_video_title(url)
    safe_title = slugify(title)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{safe_title}_summary.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    prompt = f"""
    Concisely summarize the following podcast transcript into key insights and themes.
    Write the summary and insights in one clear paragraph, then include 5 bullet-point takeaways.
    The summary must be under 1500 characters.

    Transcript:
    {transcript}  # optional limit for very long transcripts
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert and concise podcast summarizer."},
            {"role": "user", "content": prompt},
        ],
    )

    summary = response.choices[0].message.content.strip()

    # ✅ Ensure the summary is <1500 characters
    if len(summary) > 1500:
        summary = summary[:1497] + "..."

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"✅ Summary ({len(summary)} chars) saved to {output_path}")
    return output_path, title



def write_summary_to_notion(title, summary, link=None):
    NOTION_TOKEN = os.getenv("NOTION_TOKEN")
    NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
    """Write a podcast summary to your Notion database."""

    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    # Construct payload based on your exact schema
    data = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Podcast Title": {
                "title": [{"text": {"content": title}}]
            },
            "Date Added": {
                "date": {"start": datetime.now(timezone.utc).isoformat()}
            },
            "Link": {
                "url": link or ""
            },
            "Status": {
                "select": {"name": "To Listen"}  # optional default
            }
        },
        # Summary goes into the page body
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"text": {"content": summary}}
                    ]
                }
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print(f"✅ Successfully added '{title}' to Notion!")
    else:
        print("❌ Error adding to Notion:", response.status_code)
        print(response.text)