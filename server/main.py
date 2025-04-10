from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sys
from pathlib import Path
import traceback
import json
import importlib

with open("../config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

sys.path.append(config["root_path"])

from dataset import QueryDataset
from pgvector_database import PGVectorDB
from product_matching import ImageRetrieval

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_image_embedding_model_from_path(model_name: str, cfg: dict):
    model_cfg = cfg["model"][model_name]
    module = importlib.import_module(model_cfg["model_dir"])
    class_name = model_cfg["model_name"]
    cls = getattr(module, class_name)

    return cls(model_name, cfg)


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
        dataset = QueryDataset("test", config)
        dataset.clean_query_images()
        await dataset.save_query_images(file)
        query_image, query_id = dataset.prepare_query_images(0, 1)

        database = PGVectorDB(model_name, config)
        model = load_image_embedding_model_from_path(model_name, config)

        retrieval_model = ImageRetrieval(model, database, config)
        retrieval_result = retrieval_model(query_image, query_id, category1, category2)

        top_k_paths = retrieval_result['result_paths'][0] if retrieval_result['result_paths'] else []
        top_k_distances = retrieval_result['result_distances'][0]

        return JSONResponse(content={
            "similar_images": top_k_paths,
            "distances": top_k_distances
        })

    except Exception as e:
        print(f"❌ 처리 실패: {e}")
        print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


project_root = config["root_path"]
frontend_build_path = Path(project_root) / "server" / "aisum-ui" / "build"

app.mount("/static", StaticFiles(directory=frontend_build_path / "static"), name="static")
app.mount(
    f"{project_root}/outputs",
    StaticFiles(directory=Path(project_root) / "outputs"),
    name="outputs"
)
app.mount("/", StaticFiles(directory=frontend_build_path, html=True), name="frontend")


# SPA 라우팅
@app.get("/{full_path:path}")
async def serve_spa():
    return FileResponse(frontend_build_path / "index.html")
