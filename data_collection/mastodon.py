import requests
import re
import json
import os
from fast_langdetect import detect
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from huggingface_hub import HfApi, create_repo, hf_hub_download

COMPANY=os.getenv("COMPANY")
if COMPANY == None:
    raise NameError("The company has not been specified")
INSTANCE = "mastodon.social"
HASHTAGS = [COMPANY]  
SPACE_URL = os.getenv("SPACE_URL") 
if SPACE_URL == None:
    raise NameError("The domain has not been specified")
HF_USER = os.getenv("HF_USER")
if HF_USER == None:
    raise NameError("The HF USER has not been specified")
APP_URL = SPACE_URL + "/predict"

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

print(f"Collected Posts: {len(unique_posts)}")

def send_post(text):
    try:
        response = requests.post(APP_URL, json={"text": text}, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

with ThreadPoolExecutor(max_workers=5) as executor:
    predictions = list(executor.map(send_post, [post["text"] for post in unique_posts]))

for post, pred in zip(unique_posts, predictions):
    if pred:
        post["prediction"] = pred.get("sentiment")
        post["confidence"] = pred.get("confidence")

try:
    existing_path = hf_hub_download(repo_id=f"{HF_USER}/sentiment_posts", filename="new_posts.json", repo_type="dataset")
    existing_posts = json.load(open(existing_path))
except Exception:
    existing_posts = []

seen = {p["text"] for p in existing_posts}
merged_posts = existing_posts + [p for p in unique_posts if p["text"] not in seen]

with open("./new_posts.json", "w", encoding="utf-8") as f:
    json.dump(merged_posts, f, indent=2, ensure_ascii=False)


api = HfApi(token=os.getenv("HF_TOKEN"))
api.create_repo(repo_id=f"{HF_USER}/sentiment_posts", repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj="./new_posts.json",
    path_in_repo="new_posts.json",
    repo_id=f"{HF_USER}/sentiment_posts",
    repo_type="dataset"
)



