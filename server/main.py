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
from yolo import YOLO
from repository import ImageRetrievalRepository
from image_retrieval import ImageRetrieval
from aisum_database import AisumDBAdapter, get_p_key_from_result_id, deduplicate_results_per_object, get_image_urls_from_mysql

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
        dataset = QueryDataset("server", config)
        dataset.clean_query_images()
        await dataset.save_query_images(file)
        query_image, query_id = dataset.prepare_query_images(0, 1)

        # 객체탐지
        detection_model = YOLO("yolo", config)
        detection_result = detection_model(query_image, query_id)
        detection_images = detection_result["detection_images"]
        detection_ids = detection_result["original_image_ids"]
        detection_segment_ids = detection_result["image_segment_ids"]
        detection_classes = detection_result["detection_classes"]
        detection_coordinates = detection_result["detection_coordinates"]
        detection_sizes = detection_result["detection_sizes"]
        detection_centrality = detection_result["detection_centrality"]

        # 탐지된 객체 없는 경우 전체이미지로 검색
        if not detection_images:
            detection_images = query_image
            detection_segment_ids = query_id
            detection_classes = [""]
            detection_sizes = [1.0]
            detection_centrality = [1.0]

        all_similar_images = [] 
        all_p_scores = []

        # 탐지된 객체에 대한 검색
        for det_image, seg_id, det_class, det_size, det_centrality in zip(
            detection_images, detection_segment_ids, detection_classes, 
            detection_sizes, detection_centrality):
            if embedding_model_name == "ensemble":
                # 앙상블 검색
                result = repository.ensemble(
                    [det_image], [seg_id], [det_class], [det_size], [det_centrality]
                )
                if result.get('result_local_paths') and not result['result_local_paths'][0]:
                    url_results = AisumDBAdapter.get_image_urls_for_server(config, result['result_ids'])
                    result['result_local_paths'] = url_results
            else:
                # 단일 모델 검색
                embedding_model = repository.models.get_model_by_name(embedding_model_name)
                database = repository.databases.get_db_by_name(embedding_model_name)
                    
                retrieval_model = ImageRetrieval(embedding_model, database, config)
                retrieval_model.set_dataset_mode("server")
                    
                result = retrieval_model(
                        [det_image], [seg_id], [det_class], [det_size], [det_centrality]
                    )
            # 결과 처리
            if result.get('result_local_paths') and result['result_local_paths'][0]:
                similar_images = result['result_local_paths'][0]
                p_scores = result.get('p_scores', result.get('similarities', []))[0]
                result_ids = result['result_ids'][0]
                similarities = result['similarities'][0]
                
                # 객체별 중복 제거
                dedup_result_ids, dedup_similarities, dedup_p_scores = deduplicate_results_per_object(
                    result_ids, similarities, p_scores
                )
                
                # 중복 제거된 결과로 URL 조회
                if embedding_model_name == "ensemble" or retrieval_model.dataset_mode == "server":
                    # p_key 추출 후 URL 조회
                    p_keys = [get_p_key_from_result_id(rid) for rid in dedup_result_ids]
                    dedup_similar_images = get_image_urls_from_mysql(config, p_keys)
                else:
                    # 로컬 파일 경로의 경우
                    dedup_similar_images = similar_images[:len(dedup_result_ids)]
                
                all_similar_images.extend(dedup_similar_images)
                all_p_scores.extend(dedup_p_scores)


        return JSONResponse(content={
                "similar_images": all_similar_images,
                "p_scores": all_p_scores
            })

        '''
        if embedding_model_name == "ensemble":
            # 앙상블 검색
            result = repository.ensemble(query_image, query_id, "")
            distances = [1 - score for score in result['p_scores'][0]]
            return JSONResponse(content={
                "similar_images": result['result_local_paths'][0] if result['result_local_paths'] else [],
                "distances": distances
            })
        else:
            # 단일 모델 검색
            result = repository.get_retrieval_result_by_name(embedding_model_name, query_image, query_id, "")
            distances = [1 - score for score in result['p_scores'][0]]
            return JSONResponse(content={
                "similar_images": result['result_local_paths'][0] if result['result_local_paths'] else [],
                "distances": distances
            })
        '''

    except Exception as e:
        print(f"❌ 처리 실패: {e}")
        print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/detect/")
async def detect_fashion_objects(file: UploadFile = File(...)):
    try:
        dataset = QueryDataset("server", config)
        dataset.clean_query_images()
        await dataset.save_query_images(file)
        query_image, query_id = dataset.prepare_query_images(0, 1)

        detection_model = YOLO("yolo", config)
        detection_result = detection_model(query_image, query_id)
        detection_classes = detection_result["detection_classes"]
        detection_coordinates = detection_result["detection_coordinates"]
        detection_confidences = detection_result["detection_confidences"]

        detections = []
        for cls, crds, conf in zip(detection_classes, detection_coordinates, detection_confidences):
            
            detections.append({
                "class": cls,
                "bbox": crds,
                "confidence": conf
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
        dataset = QueryDataset("server", config)
        dataset.clean_query_images()
        await dataset.save_query_images(file)
        query_image, query_id = dataset.prepare_query_images(0, 1)

        cropped_image = query_image[0].crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))

        # bbox size, centrality 계산
        img_width, img_height = query_image[0].size
        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin
        bbox_size = (bbox_width * bbox_height) / (img_width * img_height)
        center_x = (bbox_xmin + bbox_xmax) / 2 / img_width
        center_y = (bbox_ymin + bbox_ymax) / 2 / img_height
        bbox_centrality = 1.0 - (abs(center_x - 0.5) + abs(center_y - 0.5))

        if model_name == "ensemble":
            # 앙상블 검색
            result = repository.ensemble(
                [cropped_image], [query_id[0]], [category or ""], [bbox_size], [bbox_centrality]
            )
            # 앙상블 결과에 대해서는 별도로 URL 조회 필요
            if result.get('result_local_paths') and not result['result_local_paths'][0]:
                url_results = AisumDBAdapter.get_image_urls_for_server(config, result['result_ids'])
                result['result_local_paths'] = url_results
        else:
            # 단일 모델 검색 - dataset mode를 server로 설정
            embedding_model = repository.models.get_model_by_name(model_name)
            database = repository.databases.get_db_by_name(model_name)
            
            retrieval_model = ImageRetrieval(embedding_model, database, config)
            retrieval_model.set_dataset_mode("server")
            
            result = retrieval_model(
                [cropped_image], [query_id[0]], [category or ""], [bbox_size], [bbox_centrality]
            )

        p_scores = result.get('p_scores', result.get('similarities', []))[0]
        similar_images = result['result_local_paths'][0] if result['result_local_paths'] else []
        result_ids = result['result_ids'][0]
        similarities = result['similarities'][0]
        
        # 객체별 중복 제거
        dedup_result_ids, dedup_similarities, dedup_p_scores = deduplicate_results_per_object(
            result_ids, similarities, p_scores
        )
        
        # 중복 제거된 결과로 URL 조회
        if model_name == "ensemble" or retrieval_model.dataset_mode == "server":
            # p_key 추출 후 URL 조회
            p_keys = [get_p_key_from_result_id(rid) for rid in dedup_result_ids]
            dedup_similar_images = get_image_urls_from_mysql(config, p_keys)
        else:
            # 로컬 파일 경로의 경우
            dedup_similar_images = similar_images[:len(dedup_result_ids)]
        
        # p_score 기준으로 정렬
        image_score_pairs = list(zip(dedup_similar_images, dedup_p_scores))
        image_score_pairs.sort(key=lambda x: x[1], reverse=True) 
        sorted_images = [pair[0] for pair in image_score_pairs]
        sorted_p_scores = [pair[1] for pair in image_score_pairs]

        return JSONResponse(content={
            "similar_images": sorted_images,
            "p_scores": sorted_p_scores
        })
        '''
        if model_name == "ensemble":
            # 앙상블 검색
            result = repository.ensemble(cropped_image, query_id, category)
            distances = [1 - score for score in result['p_scores'][0]]
            return JSONResponse(content={
                "similar_images": result['result_local_paths'][0] if result['result_local_paths'] else [],
                "distances": distances
            })
        else:
            # 단일 모델 검색
            result = repository.get_retrieval_result_by_name(model_name, cropped_image, query_id, category)
            distances = [1 - score for score in result['p_scores'][0]]
            return JSONResponse(content={
                "similar_images": result['result_local_paths'][0] if result['result_local_paths'] else [],
                "distances": distances
            })
        '''

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

output_path = Path(project_root) / "outputs"
if output_path.exists():
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
