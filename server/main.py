from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sys
from pathlib import Path
import traceback
import json
import importlib
from PIL.Image import Image

with open("../config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

sys.path.append(config["root_path"])

from dataset import QueryDataset
from pgvector_database import PGVectorDB
from product_matching import ImageRetrieval
from ensemble_retrieval import Ensemble
from repository import ModelRepository

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repository = ModelRepository(config)


@app.post("/search/")
async def search_by_original_image(
        file: UploadFile = File(...),
        embedding_model_name: str = Form(...)
):
    try:
        dataset = QueryDataset("test", config)
        dataset.clean_query_images()
        await dataset.save_query_images(file)
        query_image, query_id = dataset.prepare_query_images(0, 1)

        if embedding_model_name == "ensemble":
            # 앙상블 검색
            ensemble_model_names = config["ensemble"].values()
            retrieval_models = dict()

            for name in ensemble_model_names:
                db = PGVectorDB(name, config)
                mdl = repository.get_model_by_name(name)
                retrieval_models[name] = ImageRetrieval(mdl, db, config)

            ensemble_model = Ensemble(retrieval_models)
            result = ensemble_model(query_image, query_id, None)

            return JSONResponse(content={
                "similar_images": result['result_paths'][0] if result['result_paths'] else [],
                "distances": result['result_distances'][0]
            })
        else:
            # 단일 모델 검색
            database = PGVectorDB(embedding_model_name, config)
            embedding_model = repository.get_model_by_name(embedding_model_name)
            retrieval_model = ImageRetrieval(embedding_model, database, config)
            result = retrieval_model(query_image, query_id, None)

            return JSONResponse(content={
                "similar_images": result['result_paths'][0] if result['result_paths'] else [],
                "distances": result['result_distances'][0]
            })

    except Exception as e:
        print(f"❌ 처리 실패: {e}")
        print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/detect/")
async def detect_fashion_objects(file: UploadFile = File(...)):
    try:
        dataset = QueryDataset("test", config)
        dataset.clean_query_images()
        await dataset.save_query_images(file)
        query_image, query_id = dataset.prepare_query_images(0, 1)

        detection_model = repository.get_model_by_name("yolo")
        detection_result = detection_model(query_image, query_id)
        detection_classes = detection_result["detection_classes"]
        detection_coordinates = detection_result["detection_coordinates"]

        detections = []
        for cls, crds in zip(detection_classes, detection_coordinates):
            detections.append({
                "class": cls,
                "bbox": crds
            })

        return JSONResponse(content={
            "detections": detections
        })

    except Exception as e:
        print(f"❌ Detection 실패: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/search_bbox/")
async def search_by_bbox(file: UploadFile = File(...),
                         model_name: str = Form(...),
                         bbox_xmin: int = Form(...),
                         bbox_ymin: int = Form(...),
                         bbox_xmax: int = Form(...),
                         bbox_ymax: int = Form(...),
                         category: str = Form(None)):
    try:
        dataset = QueryDataset("test", config)
        dataset.clean_query_images()
        await dataset.save_query_images(file)
        query_image, query_id = dataset.prepare_query_images(0, 1)

        cropped_image = query_image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))

        database = PGVectorDB(model_name, config)
        embedding_model = repository.get_model_by_name(model_name)
        retrieval_model = ImageRetrieval(embedding_model, database, config)

        retrieval_result = retrieval_model(cropped_image, query_id, category)

        top_k_paths = retrieval_result['result_paths'][0] if retrieval_result['result_paths'] else []
        top_k_distances = retrieval_result['result_distances'][0]

        return JSONResponse(content={
            "similar_images": top_k_paths,
            "distances": top_k_distances
        })

    except Exception as e:
        print(f"❌ BoundingBox 검색 실패: {e}")
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


@app.get("/reset/")
async def clear_repository():
    repository.clear()
    return {"message": "Repository has been cleared."}


# SPA 라우팅
@app.get("/{full_path:path}")
async def serve_spa():
    return FileResponse(frontend_build_path / "index.html")
