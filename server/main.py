from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sys
from pathlib import Path
import traceback
import json

with open("../config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

sys.path.append(config["root_path"])

from dataset import QueryDataset
from pgvector_database import PGVectorDB
from image_retrieval import ImageRetrieval
from yolo import YOLO
from ensemble_retrieval import Ensemble
from repository import ImageRetrievalRepository

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repository = ImageRetrievalRepository(config)


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
            ensemble_result = repository.ensemble(query_image, query_id, None)

            return JSONResponse(content={
                "similar_images": ensemble_result['result_paths'][0] if ensemble_result['result_paths'] else [],
                "distances": ensemble_result['result_distances'][0]
            })
        else:
            # 단일 모델 검색
            result = repository.get_retrieval_result_by_name(embedding_model_name, query_image, query_id, None)

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

        detection_model = YOLO("yolo", config)
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

        cropped_image = query_image[0].crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))

        if model_name == "ensemble":
            # 앙상블 검색
            ensemble_result = repository.ensemble(cropped_image, query_id, category)

            return JSONResponse(content={
                "similar_images": ensemble_result['result_paths'][0] if ensemble_result['result_paths'] else [],
                "distances": ensemble_result['result_distances'][0]
            })
        else:
            # 단일 모델 검색
            result = repository.get_retrieval_result_by_name(model_name, cropped_image, query_id, category)

            return JSONResponse(content={
                "similar_images": result['result_paths'][0] if result['result_paths'] else [],
                "distances": result['result_distances'][0]
            })

    except Exception as e:
        print(f"❌ BoundingBox 검색 실패: {e}")
        print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/reset")
async def reset():
    repository.clear_retrieval_results()
    return {"message": "Retrieval results has been cleared."}


@app.get("/reset_all")
async def reset_all():
    repository.reset()
    return {"message": "Repository has been cleared."}


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
