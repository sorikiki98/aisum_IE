from collections import defaultdict
from typing import List, Dict, Tuple
import torch.nn as nn
from product_matching import ImageRetrieval
import os
import shutil
from PIL import Image


class EnsembleImageRetrieval(nn.Module):
    def __init__(self, models: Dict[str, ImageRetrieval]):
        super().__init__()
        self.models = models
        self.model_weights = {
            "magiclens": [x * 0.5 * 0.5 for x in range(10, 0, -1)],
            "dreamsim": [x * 0.5 for x in range(10, 0, -1)],
            "marqo_ecommerce_l": [x * 0.5 * 0.75 for x in range(10, 0, -1)]
        }
        first_model = next(iter(models.values()))
        self.retrieved_image_folder = first_model.retrieved_image_folder
        self.index_image_folder = first_model.index_image_folder
        self.model_name = "ensemble"

    def forward(self, query_image, query_id, cat1_code=None, cat2_code=None):
        # 각 모델별 검색 결과 수집
        model_results = {}
        for model_name, model in self.models.items():
            result = model(query_image, query_id, cat1_code, cat2_code)
            model_results[model_name] = result

        # 이미지별 점수 집계
        image_scores = defaultdict(float)
        image_paths = {}  # 이미지 경로 저장

        for model_name, result in model_results.items():
            result_ids = result["result_ids"][0]  # 첫 번째 배치만 사용
            result_paths = result["result_paths"][0]  # 경로 정보 가져오기
            weights = self.model_weights[model_name]

            for rank, (img_id, img_path) in enumerate(zip(result_ids, result_paths)):
                if rank < len(weights):  # top 10까지만 점수 부여
                    image_scores[img_id] += weights[rank]
                    if img_id not in image_paths:
                        image_paths[img_id] = img_path

        # 점수 기준 상위 10개 이미지 선정
        top_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # 결과 저장
        result_ids = []
        result_similarities = []
        retrieved_image_file_paths = []

        # 앙상블 결과 이미지 저장
        ensemble_folder = os.path.join(self.retrieved_image_folder, self.model_name, "0")
        if os.path.exists(ensemble_folder):
            shutil.rmtree(ensemble_folder)
        os.makedirs(ensemble_folder, exist_ok=True)

        batch_paths = []
        for idx, (img_id, score) in enumerate(top_images):
            result_ids.append(img_id)
            result_similarities.append(float(score))

            # 원본 이미지 경로 사용
            original_path = image_paths.get(img_id)

            if original_path and os.path.exists(original_path):
                save_name = f"top_{idx + 1}_{score}.jpg"
                save_path = os.path.join(ensemble_folder, save_name)
                image = Image.open(original_path)
                image.save(save_path)
                relative_path = f"{ensemble_folder}/{save_name}"
                batch_paths.append(relative_path)
            else:
                print(f"Image not found for ID: {img_id}")
                batch_paths.append(None)

        retrieved_image_file_paths.append(batch_paths)

        return {
            "result_ids": [result_ids],
            "result_distances": [result_similarities],
            "result_paths": retrieved_image_file_paths
        }
