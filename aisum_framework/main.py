from fastapi import FastAPI, UploadFile, File, Form
from fastapi.routing import APIRoute
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sys
from pathlib import Path

# 🔁 경로 등록
sys.path.append(str(Path(__file__).resolve().parents[1]))

from image_embedding_model import get_image_embedding
from vector_database import search_similar_images

app = FastAPI()

# ✅ CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ /embed API (카테고리 필터 추가 버전)
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
        image_bytes = await file.read()
        query_vector = get_image_embedding(image_bytes, model_name)
    except Exception as e:
        print(f"❌ 모델 임베딩 실패: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

    try:
        top_k_paths = search_similar_images(
            query_vector,
            model_name,
            top_k=10,
            category1=category1,
            category2=category2
        )
        return JSONResponse(content={"similar_images": top_k_paths})
    except Exception as e:
        print(f"❌ 유사 이미지 검색 실패: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ✅ build 폴더 경로를 먼저 정의
frontend_build_path = Path(__file__).resolve().parent / "aisum-ui" / "build"

# ✅ SPA 정적 파일 서빙
app.mount("/", StaticFiles(directory=frontend_build_path, html=True), name="frontend")

# ✅ 이미지 경로 서빙
app.mount("/images", StaticFiles(directory=Path(__file__).resolve().parents[1] / "images"), name="images")

# ✅ SPA 라우팅 (index.html 반환)
@app.get("/{full_path:path}")
async def serve_spa():
    return FileResponse(frontend_build_path / "index.html")