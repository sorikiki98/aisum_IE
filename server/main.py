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

from product_matching import main as product_matching_main

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
    category1: str = Form(None),
    category2: str = Form(None),
):
    print("📥 /embed/ POST 요청 도착")
    print(f"✅ 업로드된 파일 이름: {file.filename}")
    print(f"[FILTER] category1: {category1}, category2: {category2}")
    print(f"[DEBUG] 전달된 model_name: {model_name}")

    try:
        # Clean up test/images directory
        project_root = Path(__file__).resolve().parent.parent
        query_save_dir = project_root / "data" / "test" / "images"
        if query_save_dir.exists():
            for image_file in query_save_dir.glob("*"):
                try:
                    image_file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete {image_file}: {e}")
        query_save_dir.mkdir(parents=True, exist_ok=True)

        # Read image bytes only once
        image_bytes = await file.read()
        
        # Verify image bytes are not empty
        if not image_bytes:
            raise ValueError("Uploaded file is empty")

        # Save query image to data/test/images
        extension = Path(file.filename).suffix
        query_filename = f"{uuid4().hex}{extension}"
        query_save_path = query_save_dir / query_filename

        with open(query_save_path, "wb") as f:
            f.write(image_bytes)

        try:
            from PIL import Image
            Image.open(query_save_path).verify()
        except Exception as e:
            raise ValueError(f"Saved image is invalid: {e}")
        
        result = product_matching_main(
            model_name=model_name,
            category1=category1,
            category2=category2
        )

        # 현재는 첫 번째 쿼리 이미지의 결과만 사용
        top_k_paths = result['result_paths'][0] if result['result_paths'] else []
        top_k_distances = result['result_distances'][0]

        return JSONResponse(content={
            "similar_images": top_k_paths,
            "distances": top_k_distances
        })

    except Exception as e:
        print(f"❌ 처리 실패: {e}")
        print(traceback.format_exc())
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
