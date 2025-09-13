import os
import json
from fastapi import FastAPI, Form

app = FastAPI()

RESULT_DIR = "./results"   # JSON 저장 폴더


@app.post("/process")
async def process_input(
    image_url: str = Form(...),   # 이미지 파일
    k_per_object: int = Form(...),              # 탐지 개수
    category_filter: bool = Form(...)       # 카테고리 유무
):
    # 결과 디렉토리 생성
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    #JSON 파일명
    existing = [f for f in os.listdir(RESULT_DIR) if f.startswith("query_image") and f.endswith(".json")]
    next_idx = len(existing) + 1
    json_filename = f"query_image{next_idx}.json"
    json_path = os.path.join(RESULT_DIR, json_filename)

    # 결과 딕셔너리
    result = {
        "image_url": image_url,
        "k_per_object": k_per_object,
        "category_filter": category_filter
    }

    # 결과를 JSON 파일로 저장 
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print("서버 로그:", result, flush=True)  # 로그 출력

    return {
        "json_file": json_filename,
        "result": result
    }