import requests
import re
import json
import os
from fast_langdetect import detect
from concurrent.futures import ThreadPoolExecutor
from html import unescape

INSTANCE = "mastodon.social"
HASHTAGS = ["Anthropic", "claude"]  
APP_URL = os.getenv("APP_PRED_URL")


def get_language(text: str) -> str:
    return detect(text, model="auto", k=1)[0]["lang"]

def clean_html(html: str) -> str:
    clean = unescape(html)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = re.sub(r"http\S+", "", clean)
    clean = re.sub(r".com\S+", "", clean)
    clean = re.sub(r'/\S+', "", clean)
    clean = re.sub(r"#\w+", "", clean)
    clean = re.sub(r"@\w+", "", clean)
    clean = re.sub(r"&\w+;", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def get_post(hashtag: str, limit: int = 40, backtracking: int = 5) -> list:
    url = f"https://{INSTANCE}/api/v1/timelines/tag/{hashtag}"
    max_id = None
    result = []
    for i in range(backtracking):
        params = {"limit": limit}
        if max_id:
            params["max_id"] = max_id

        response = requests.get(url=url, params=params)
        posts = response.json()

        if not posts:
            break

        
        for p in posts:
            text = clean_html(p["content"])
            if len(text.split()) < 5:
                continue
            if get_language(text) != "en":
                continue

            result.append({
                "text": text,
                "likes": p["favourites_count"],
                "replies_count": p["replies_count"],
                "quotes_count": p["quotes_count"],
                "reblogs_count": p["reblogs_count"],
            })
        max_id = posts[-1]["id"]
    return result


all_posts = []

for hashtag in HASHTAGS:
    all_posts.extend(get_post(hashtag, limit=40))

seen = set()
unique_posts = []
for post in all_posts:
    if post["text"] not in seen:
        seen.add(post["text"])
        unique_posts.append(post)

print(f"Post raccolti: {len(unique_posts)}")

with open("./new_posts.json", "w", encoding="utf-8") as f:
    json.dump(unique_posts, f, indent=2, ensure_ascii=False)

texts = [post["text"] for post in unique_posts]
def send_post(text):
    return requests.post(APP_URL, json={"text": text}, timeout=15)

with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(send_post, texts)
