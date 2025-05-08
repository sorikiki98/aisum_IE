from tqdm import tqdm
import numpy as np
import os
from PIL import Image, UnidentifiedImageError
from pathlib import Path


class IndexDataset:
    def __init__(self, dataset_name, config):
        self.dataset_name = dataset_name

        index_image_folder = Path(config["data"]["index_image_folder_path"])
        image_extensions = config["data"]["image_extensions"]
        index_image_files = sorted(
            file for file in index_image_folder.rglob("*")
            if file.suffix.lower() in image_extensions
        )
        index_image_ids = [file.parent.name + "_" + file.stem for file in index_image_files]
        self.index_image_files = index_image_files
        self.index_image_ids = index_image_ids

    def prepare_index_images(self, batch_idx, batch_size):
        batch_files = self.index_image_files[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_ids = self.index_image_ids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_ids = np.array(batch_ids)
        batch_images = []

        for file_path, img_id in zip(batch_files, batch_ids):
            try:
                img = Image.open(file_path).convert("RGB")
                batch_images.append(img)
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                continue
        return batch_images, batch_ids

    def truncate_index_images(self, row_count=0):
        self.index_image_files = self.index_image_files[row_count:]
        self.index_image_ids = self.index_image_ids[row_count:]


class QueryDataset:
    def __init__(self, dataset_name, config):
        self.dataset_name = dataset_name

        query_image_folder = Path(config["data"]["query_image_folder_path"])
        query_image_files = sorted(query_image_folder.glob("**/*.jpg"))
        query_image_ids = [file.stem for file in query_image_files]
        self.query_image_folder = query_image_folder
        self.query_image_files = query_image_files
        self.query_image_ids = query_image_ids

    async def save_query_images(self, image_file):
        query_filename = "0.jpg"  # todo
        image_bytes = await image_file.read()

        if not image_bytes:
            raise ValueError("Uploaded file is empty")

        query_image_file_path = Path(os.path.join(str(self.query_image_folder), str(query_filename)))
        query_image_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(Path(query_image_file_path), "wb") as f:
            f.write(image_bytes)

    def clean_query_images(self):
        if self.query_image_folder.exists() and self.query_image_folder.is_dir():
            for file in self.query_image_folder.glob("*"):
                if file.is_file():
                    file.unlink()

    def prepare_query_images(self, batch_idx, batch_size):
        batch_files = self.query_image_files[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_ids = self.query_image_ids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_ids = np.array(batch_ids)
        batch_images = []

        for file_path, img_id in zip(batch_files, batch_ids):
            img = Image.open(file_path).convert("RGB")
            batch_images.append(img)
        return batch_images, batch_ids
