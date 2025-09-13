import os
import json

RESULT_DIR = "./results"

def load_all_query(query: dict):
    for f in os.listdir(RESULT_DIR):
        if f.startswith("query_image") and f.endswith(".json"):
            base_name = os.path.splitext(f)[0]   # query_image1.json → query_image1
            if base_name not in query:  # 이미 있는 건 건너뛰기
                json_path = os.path.join(RESULT_DIR, f)
                with open(json_path, "r", encoding="utf-8") as jf:
                    query[base_name] = json.load(jf)
    return query

query = load_all_query({})