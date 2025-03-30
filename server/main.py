from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sys
from pathlib import Path
from uuid import uuid4


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

        project_root = Path(__file__).parent.parent
        save_dir = project_root / "output" / model_name / category1 / category2
        save_dir.mkdir(parents=True, exist_ok=True)

        extension = Path(file.filename).suffix
        filename = f"{uuid4().hex}{extension}"
        save_path = save_dir / filename

        with open(save_path, "wb") as f:
            f.write(image_bytes)

        with open(save_path, "rb") as f:
            image_bytes = f.read()

        model, model_params = load_image_embedding_model(model_name)

        query_vector = embed_images(
            image_embedding_model=model,
            image_embedding_model_name=model_name,
            model_params=model_params,
            query_image_bytes=[image_bytes]
        )[0]

    except Exception as e:
        print(f"❌ 모델 임베딩 실패: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

    try:
        result_ids, _, _, result_distances = search_similar_vectors(
            image_embedding_model_name=model_name,
            query_embeddings=[query_vector],
            query_ids=["uploaded_query"],
            category1=category1,
            category2=category2,
        )

        # ✅ 복사 실행
        save_retrieved_images_by_ids(
            image_embedding_model_name=model_name,
            all_batch_ids=result_ids,
            all_batch_similarities=result_distances,
            category1=category1,
            category2=category2
        )

        # ✅ 프론트용 경로 리턴
        top_k_paths = [f"images/{category1}/{category2}/{img_id}.jpg" for img_id in result_ids[0]]
        top_k_distances = result_distances[0]

        return JSONResponse(content={
            "similar_images": top_k_paths,
            "distances": top_k_distances
        })

    except Exception as e:
        print(f"❌ 유사 이미지 검색 실패: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ✅ 정적 파일 경로: 프론트엔드 빌드
frontend_build_path = Path(__file__).resolve().parent / "aisum-ui" / "build"
app.mount("/static", StaticFiles(directory=frontend_build_path / "static"), name="static")

app.mount(
    "/images",
    StaticFiles(directory=Path(__file__).parent.parent / "data" / "eseltree" / "images"),
    name="images"
)

app.mount("/", StaticFiles(directory=frontend_build_path, html=True), name="frontend")


# ✅ SPA 라우팅
@app.get("/{full_path:path}")
async def serve_spa():
    return FileResponse(frontend_build_path / "index.html")
