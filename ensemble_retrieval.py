from collections import defaultdict
from pathlib import Path
import torch.nn as nn
import os
import shutil
from PIL import Image


class Ensemble(nn.Module):
    def __init__(self, retrieval_results, config):
        super().__init__()
        self.retrieval_results = retrieval_results
        self.retrieved_image_folder = config["data"]["retrieved_image_folder_path"]
        self.index_image_folder = Path(config["data"]["index_image_folder_path"])
        self.model_name = "ensemble"

    def forward(self):
        first_key, first_value = next(iter(self.retrieval_results.items()))
        n_objects = len(first_value["result_ids"])

        top_images_per_obj = []
        image_paths = {}  # 이미지 경로 저장
        image_cats = {}  # 이미지 카테고리 저장
        for obj_i in range(n_objects):
            # 이미지별 점수 집계
            image_scores = defaultdict(float)

            for i, (model_name, result) in enumerate(self.retrieval_results.items()):
                if i == 0:
                    model_weight = 0.65
                else:
                    model_weight = 0.35
                result_ids = result["result_ids"][obj_i]
                # result_paths = result["result_local_paths"][obj_i]
                result_cats = result["result_categories"][obj_i]
                similarities = result["similarities"][obj_i]
                '''
                for rank, (img_id, img_path, cat, p_score) in enumerate(
                        zip(result_ids, result_paths, result_cats, p_scores)):
                    image_scores[img_id] += p_score * model_weight
                    if img_id not in image_paths:
                        image_paths[img_id] = img_path
                    if img_id not in image_cats:
                        image_cats[img_id] = cat
                '''
                for rank, (img_id, cat, similarity) in enumerate(
                        zip(result_ids, result_cats, similarities)):
                    image_scores[img_id] += similarity * model_weight
                    if img_id not in image_cats:
                        image_cats[img_id] = cat
            # 점수 기준 상위 30개 이미지 선정
            top_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)[:30]
            top_images_per_obj.append(top_images)

        # 앙상블 결과 이미지 저장
        ensemble_folder = os.path.join(self.retrieved_image_folder, self.model_name, "0")
        if os.path.exists(ensemble_folder):
            shutil.rmtree(ensemble_folder)
        os.makedirs(ensemble_folder, exist_ok=True)

        # 결과 저장
        result_ids = []
        retrieved_image_file_paths = []
        retrieved_categories = []
        similarities = []

        for top_images in top_images_per_obj:  # 하나의 객체
            result_ids_per_obj = []
            retrieved_image_file_paths_per_obj = []
            retrieved_categories_per_obj = []
            similarities_per_obj = []
            for idx, (img_id, score) in enumerate(top_images):  # 하나의 이미지
                result_ids_per_obj.append(img_id)
                similarities_per_obj.append(score)
                cat = image_cats.get(img_id)
                retrieved_categories_per_obj.append(cat)

                '''
                img_path = image_paths.get(img_id)
                if img_path and os.path.exists(img_path):
                    save_name = f"top_{idx + 1}_{score}.jpg"
                    save_path = os.path.join(ensemble_folder, save_name)
                    image = Image.open(img_path)
                    image.save(save_path)
                    relative_path = f"{ensemble_folder}/{save_name}"
                    retrieved_image_file_paths_per_obj.append(relative_path)
                else:
                    print(f"Image not found for ID: {img_id}")
                    retrieved_image_file_paths_per_obj.append(None)
                '''

            result_ids.append(result_ids_per_obj)
            # retrieved_image_file_paths.append(retrieved_image_file_paths_per_obj)
            retrieved_categories.append(retrieved_categories_per_obj)
            similarities.append(similarities_per_obj)

            retrieved_image_file_paths = []

        return {
            "result_ids": result_ids,
            "result_local_paths": retrieved_image_file_paths,
            "result_categories": retrieved_categories,
            "similarities": similarities
        }
