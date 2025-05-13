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
        model_weights = dict()
        for i, model_name in enumerate(retrieval_results.keys()):
            if i == 0:
                model_weights[model_name] = [x * 0.5 for x in range(10, 0, -1)]
            elif i == 1:
                model_weights[model_name] = [x * 0.5 * 0.75 for x in range(10, 0, -1)]
            else:
                model_weights[model_name] = [x * 0.5 * 0.5 for x in range(10, 0, -1)]
        self.model_weights = model_weights

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

            for model_name, result in self.retrieval_results.items():
                result_ids = result["result_ids"][obj_i]
                result_paths = result["result_local_paths"][obj_i]
                result_cats = result["result_categories"][obj_i]
                weights = self.model_weights[model_name]

                for rank, (img_id, img_path, cat) in enumerate(zip(result_ids, result_paths, result_cats)):
                    if rank < len(weights):  # top 10까지만 점수 부여
                        image_scores[img_id] += weights[rank]
                        if img_id not in image_paths:
                            image_paths[img_id] = img_path
                        if img_id not in image_cats:
                            image_cats[img_id] = cat

            # 점수 기준 상위 10개 이미지 선정
            top_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)[:10]
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
        p_scores = []

        for top_images in top_images_per_obj:  # 하나의 객체
            result_ids_per_obj = []
            retrieved_image_file_paths_per_obj = []
            retrieved_categories_per_obj = []
            p_scores_per_obj = []
            for idx, (img_id, score) in enumerate(top_images):  # 하나의 이미지
                result_ids_per_obj.append(img_id)
                p_scores_per_obj.append(float(score)/5.0)
                cat = image_cats.get(img_id)
                retrieved_categories_per_obj.append(cat)

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

            result_ids.append(result_ids_per_obj)
            retrieved_image_file_paths.append(retrieved_image_file_paths_per_obj)
            retrieved_categories.append(retrieved_categories_per_obj)
            p_scores.append(p_scores_per_obj)

        return {
            "result_ids": result_ids,
            "result_local_paths": retrieved_image_file_paths,
            "result_categories": retrieved_categories,
            "p_scores": p_scores
        }
