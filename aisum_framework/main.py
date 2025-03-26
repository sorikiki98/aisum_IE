from fastapi import FastAPI, UploadFile, File, Form
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

# ✅ 이미지 정적 경로
app.mount("/images", StaticFiles(directory=Path(__file__).resolve().parents[1] / "images"), name="images")

# ✅ React 정적 파일 빌드된 결과물 서빙
frontend_build_path = Path(__file__).resolve().parent / "aisum-ui" / "build"
app.mount("/", StaticFiles(directory=frontend_build_path, html=True), name="frontend")

# ✅ SPA 라우팅 지원 (모든 경로를 index.html로 연결)
@app.get("/{full_path:path}")
async def serve_spa():
    return FileResponse(frontend_build_path / "index.html")

# ✅ /embed API (변경 없음)
@app.post("/embed/")
async def embed_and_search_similar_images(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    try:
        image_bytes = await file.read()
        query_vector = get_image_embedding(image_bytes, model_name)
        top_k_paths = search_similar_images(query_vector, model_name, top_k=10)
        return JSONResponse(content={"similar_images": top_k_paths})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)