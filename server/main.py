from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sys
from pathlib import Path
from uuid import uuid4
import traceback
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))

from image_embedding_model import (
    load_image_embedding_model,
    embed_images,
)
from pgvector_database import search_similar_vectors
from product_matching import save_retrieved_images_by_ids

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/embed/")
async def embed_and_search_similar_images(
        file: UploadFile = File(...),
        model_name: str = Form(...),
        category1: str = Form(...),
        category2: str = Form(...),
):
    print("📥 /embed/ POST 요청 도착")
    print(f"✅ 업로드된 파일 이름: {file.filename}")
    print(f"[FILTER] category1: {category1}, category2: {category2}")
    print(f"[DEBUG] 전달된 model_name: {model_name}")

    try:
        image_bytes = await file.read()
        project_root = Path(__file__).resolve().parent.parent
        query_save_dir = project_root / "data" / "test" / "images"
        query_save_dir.mkdir(parents=True, exist_ok=True)

        extension = Path(file.filename).suffix
        query_filename = f"{uuid4().hex}{extension}"
        query_save_path = query_save_dir / query_filename

        with open(query_save_path, "wb") as f:
            f.write(image_bytes)

        from product_matching import main as product_matching_main
        result = product_matching_main(
            model_name=model_name,
            category1=category1,
            category2=category2
        )

        # Get the sorted list of result images from the output directory
        output_dir = project_root / "outputs" / model_name / "0"
        print(f"Looking for results in: {output_dir}")
        
        # Get all top_*.jpg files and sort them by number
        result_images = sorted(
            [f for f in output_dir.glob("top_*.jpg")],
            key=lambda x: int(x.stem.split('_')[1])  # Sort by the number after 'top_'
        )
        
        print(f"Found {len(result_images)} result images")
        
        # Convert paths to relative format for frontend
        top_k_paths = [f"outputs/{model_name}/0/{img.name}" for img in result_images]
        top_k_distances = result['result_distances'][0]

        return JSONResponse(content={
            "similar_images": top_k_paths,
            "distances": top_k_distances
        })

    except Exception as e:
        traceback.print_exc()
        print(f"❌ 처리 실패: {e.args}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ✅ 정적 파일 경로: 프론트엔드 빌드
frontend_build_path = Path(__file__).resolve().parent / "aisum-ui" / "build"
app.mount("/static", StaticFiles(directory=frontend_build_path / "static"), name="static")

# ✅ output 디렉토리도 정적 파일로 마운트
project_root = Path(__file__).parent.parent
app.mount(
    "/outputs",
    StaticFiles(directory=project_root / "outputs"),
    name="outputs"
)

app.mount("/", StaticFiles(directory=frontend_build_path, html=True), name="frontend")


# ✅ SPA 라우팅
@app.get("/{full_path:path}")
async def serve_spa():
    return FileResponse(frontend_build_path / "index.html")
