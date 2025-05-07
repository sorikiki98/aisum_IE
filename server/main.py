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
from ensemble_retrieval import Ensemble
from yolo import ObjectDetectionModel

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


def get_object_detection_result(pil_img):
    detection_model = ObjectDetectionModel()
    detection_output = detection_model(pil_img)[0]

    return detection_output[0], detection_output[1:]


@app.post("/search/")
async def search_by_original_image(
        file: UploadFile = File(...),
        model_name: str = Form(...),
        category: str = Form(None)
):
    try:
        dataset = QueryDataset("test", config)
        dataset.clean_query_images()
        await dataset.save_query_images(file)
        query_image, query_id = dataset.prepare_query_images(0, 1)

        if model_name == "ensemble":
            # 앙상블 검색만 실행
            models = {}
            ensemble_model_names = ["dreamsim", "magiclens", "marqo_ecommerce_l"]

            for name in ensemble_model_names:
                db = PGVectorDB(name, config)
                mdl = load_image_embedding_model_from_path(name, config)
                models[name] = ImageRetrieval(mdl, db, config)

            ensemble_model = Ensemble(models)
            result = ensemble_model(query_image, query_id, category)

            return JSONResponse(content={
                "similar_images": result['result_paths'][0] if result['result_paths'] else [],
                "distances": result['result_distances'][0]
            })
        else:
            # 단일 모델 검색
            database = PGVectorDB(model_name, config)
            model = load_image_embedding_model_from_path(model_name, config)
            retrieval_model = ImageRetrieval(model, database, config)
            result = retrieval_model(query_image, query_id, category)

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

        original_image, detected_objects = get_object_detection_result(query_image)

        result = []
        for obj in detected_objects:
            cls, bbox, _ = obj
            result.append({
                "class": cls,
                "bbox": bbox  # (xmin, ymin, xmax, ymax)
            })

        return JSONResponse(content={
            "detections": result
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
        embedding_model = load_image_embedding_model_from_path(model_name, config)
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


# SPA 라우팅
@app.get("/{full_path:path}")
async def serve_spa():
    return FileResponse(frontend_build_path / "index.html")
