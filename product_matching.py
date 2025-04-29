import torch.nn as nn
import os
import shutil
from PIL import Image


class ImageRetrieval(nn.Module):
    def __init__(self, model, database, config):
        super().__init__()
        database.print_pgvector_info()

        self._model = model
        self._database = database

        self.retrieved_image_folder = config["data"]["retrieved_image_folder_path"]
        self.index_image_folder = config["data"]["index_image_folder_path"]
        self.model_name = model.name

    def forward(self, query_image, query_id, cat1_code=None, cat2_code=None):
        query_embeddings_ndarray = self._model(query_image)
        cat1_code = cat1_code if not len(cat1_code) == 0 else None
        cat2_code = cat2_code if not len(cat2_code) == 0 else None
        result_ids, result_cat1s, result_cat2s, result_similarities = (self._database.search_similar_vectors
                                                                       (query_id,
                                                                        query_embeddings_ndarray,
                                                                        cat1_code,
                                                                        cat2_code))
        retrieved_image_file_paths = []
        for i, (result_id, result_cat1, result_cat2, result_similarity) in enumerate(
                zip(result_ids, result_cat1s, result_cat2s, result_similarities)):

            batch_paths = []
            retrieved_image_folder = os.path.join(self.retrieved_image_folder, self.model_name, str(i))

            if os.path.exists(retrieved_image_folder):
                shutil.rmtree(retrieved_image_folder)
            os.makedirs(retrieved_image_folder, exist_ok=True)

            for idx, (img_id, cat1, cat2, similarity) in enumerate(
                    zip(result_id, result_cat1, result_cat2, result_similarity)):
                file_path = os.path.join(self.index_image_folder, cat1, cat2, f"{img_id}.jpg")
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
