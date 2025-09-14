import os
import json
import requests
from PIL import Image
from io import BytesIO

RESULT_DIR = "./results"

def url_to_image(url: str):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_all_query(query: dict):
    for f in os.listdir(RESULT_DIR):
        if f.startswith("query_image") and f.endswith(".json"):
            base_name = os.path.splitext(f)[0]
            if base_name not in query:
                json_path = os.path.join(RESULT_DIR, f)
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                # URL → PIL.Image 변환
                try:
                    data["image_obj"] = url_to_image(data["image_url"])
                except Exception as e:
                    print(f"[ERROR] {base_name}: 이미지 로드 실패 ({e})")
                    data["image_obj"] = None
                query[base_name] = data
    return query

query = load_all_query({})