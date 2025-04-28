from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sys
from pathlib import Path
from uuid import uuid4
import traceback
import json
import importlib
from PIL import Image
from ultralytics import YOLO
import io



with open("../config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

sys.path.append(config["root_path"])

# 모델 로드
def load_object_detection_model():
    return YOLO('semi_lr1e4_drop04.pt')

# 중복 bounding box 제거 로직
def filter_approx_duplicate_bboxes(candidates, dif):
    sorted_c = sorted(candidates, key=lambda x: x["conf"], reverse=True)
    unique = []
    for c in sorted_c:
        xmin, ymin, xmax, ymax = c["bbox"]
        is_dup = False
        for u in unique:
            uxmin, uymin, uxmax, uymax = u["bbox"]
            if (abs(xmin - uxmin) <= dif and
                abs(ymin - uymin) <= dif and
                abs(xmax - uxmax) <= dif and
                abs(ymax - uymax) <= dif):
                is_dup = True
                break
        if not is_dup:
            unique.append(c)
    return unique

# bounding box가 가장자리 깊은 곳에 있을 경우 삭제하는 로직
def filter_edge_boxes(candidates, orig_w, orig_h):
    def is_at_border(c):
        xmin, ymin, xmax, ymax = c["bbox"]
        if (ymax < 0.02 * orig_h) and (ymin < 0.02 * orig_h):
            return True
        if (ymin > 0.80 * orig_h) and (ymax > 0.98 * orig_h):
            return True
        if (xmax < 0.02 * orig_w) and (xmin < 0.02 * orig_w):
            return True
        if (xmin > 0.80 * orig_w) and (xmax > 0.98 * orig_w):
            return True
        return False
    return [c for c in candidates if not is_at_border(c)]

# confidence 낮은 값 삭제 로직
def filter_low_confidence(candidates, min_conf=0.4):
    return [c for c in candidates if c["conf"] >= min_conf]

# top-k 박스 선택 로직
def select_topk_by_area(candidates, k=3):
    return sorted(candidates, key=lambda x: x["area"], reverse=True)[:k]

# detection 수행
def detect_objects(image: Image.Image, model):

    results = model.predict(image, verbose=False)[0]
    orig_arr = results.orig_img[:, :, ::-1]
    original_image = Image.fromarray(orig_arr)
    orig_h, orig_w = results.boxes.orig_shape

    candidates = []
    for cls_id, conf, bbox in zip(results.boxes.cls,
                                  results.boxes.conf,
                                  results.boxes.xyxy):
        xmin, ymin, xmax, ymax = map(int, bbox)
        area = (xmax - xmin) * (ymax - ymin)
        candidates.append({
            "class": results.names[int(cls_id)],
            "conf":  float(conf),
            "bbox":  (xmin, ymin, xmax, ymax),
            "area":  area,
            "image": original_image.crop((xmin, ymin, xmax, ymax))
        })

    candidates = filter_approx_duplicate_bboxes(candidates, dif=15)
    candidates = filter_edge_boxes(candidates, orig_w, orig_h)
    candidates = filter_low_confidence(candidates, min_conf=0.4)

    # 위 조건 적용하고 3개 이상일 경우 -> box 사이즈대로 3개만 출력
    if len(candidates) > 3:
        candidates = select_topk_by_area(candidates, k=3)

    return [original_image] + [
        [c["class"], c["bbox"], c["image"]] for c in candidates
    ]


from dataset import QueryDataset
from pgvector_database import PGVectorDB
from product_matching import ImageRetrieval

app = FastAPI()
detection_model = load_object_detection_model()

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

@app.post("/detect/")
async def detect_fashion_objects(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        detections = detect_objects(image, detection_model)

        original_image = detections[0]
        detection_list = detections[1:]

        result = []
        for det in detection_list:
            cls, bbox, _ = det
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
                         category1: str = Form(None),
                         category2: str = Form(None)):
    try:
        # (1) 파일 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # (2) bbox 영역으로 crop
        cropped_image = image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))

        # (3) 모델 및 DB 로드
        database = PGVectorDB(model_name, config)
        model = load_image_embedding_model_from_path(model_name, config)
        retrieval_model = ImageRetrieval(model, database, config)

        # (4) crop 이미지를 임베딩하고 검색
        query_id = ["bbox_query"]   # (id는 임시로 아무거나)
        query_image = [cropped_image]  # (Image 리스트로)

        retrieval_result = retrieval_model(query_image, query_id, category1, category2)

        # (5) 검색 결과 정리
        top_k_paths = retrieval_result['result_paths'][0] if retrieval_result['result_paths'] else []
        top_k_distances = retrieval_result['result_distances'][0]

        # (6) 결과 반환
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

# output 디렉토리도 정적 파일로 마운트
project_root = Path(__file__).parent.parent
app.mount(
    "/outputs",
    StaticFiles(directory=project_root / "outputs"),
    name="outputs"
)

app.mount("/", StaticFiles(directory=frontend_build_path, html=True), name="frontend")

# SPA 라우팅
@app.get("/{full_path:path}")
async def serve_spa():
    return FileResponse(frontend_build_path / "index.html")