import torch.nn as nn
import os
import shutil
from PIL import Image
from pathlib import Path


class ImageRetrieval(nn.Module):
    def __init__(self, model, database, config):
        super().__init__()
        database.get_pgvector_info()

        self._model = model
        self._database = database

        self.retrieved_image_folder = config["data"]["retrieved_image_folder_path"]
        self.index_image_folder = Path(config["data"]["index_image_folder_path"])
        self.model_name = model.name

    def forward(self, query_image, query_id, category):
        if not isinstance(query_image, list):
            query_image = [query_image]
        query_embeddings_ndarray = self._model(query_image)
        result_ids, result_cats, result_similarities = (self._database.search_similar_vectors
                                                        (query_id,
                                                         query_embeddings_ndarray,
                                                         category))
        retrieved_image_file_paths = []
        for i, (result_id, result_cat, result_similarity) in enumerate(
                zip(result_ids, result_cats, result_similarities)):

            batch_paths = []
            retrieved_image_folder = os.path.join(self.retrieved_image_folder, self.model_name, str(i))

            if os.path.exists(retrieved_image_folder):
                shutil.rmtree(retrieved_image_folder)
            os.makedirs(retrieved_image_folder, exist_ok=True)

            for idx, (img_id, cat, similarity) in enumerate(
                    zip(result_id, result_cat, result_similarity)):
                img_id = img_id.split("_")[1:-1]
                img_id = "_".join(img_id)
                file_path = list(self.index_image_folder.rglob(f"{img_id}.*"))[0]
                save_name = f"top_{idx + 1}_{similarity}.jpg"
                save_path = os.path.join(str(retrieved_image_folder), save_name)

                if os.path.exists(file_path):
                    image = Image.open(file_path)
                    image.save(save_path)
                    relative_path = f"{retrieved_image_folder}/{save_name}"
                    batch_paths.append(relative_path)
                else:
                    print(f"File does not exist: {file_path}")

            retrieved_image_file_paths.append(batch_paths)
        return {
            "result_ids": result_ids,
            "result_distances": result_similarities,
            "result_paths": retrieved_image_file_paths
        }
