import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from typing import Any, List, Union
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image


@dataclass
class QueryExample:
    qid: str
    qtokens: np.ndarray
    qimage: Union[np.ndarray, torch.Tensor]
    target_iid: Union[int, str, List[int], List[str], None]  # can be int or
    retrieved_iids: List[Union[int, str]]  # ranked by score, can be str (cirr) or int (circo)
    retrieved_scores: List[float]  # ranked by order


@dataclass
class IndexExample:
    iid: Union[int, str]
    iimage: Union[np.ndarray, torch.Tensor, dict]
    itokens: np.ndarray


@dataclass
class Dataset:
    name: str
    query_examples: List[QueryExample] = field(default_factory=list)
    k_range: List[int] = field(default_factory=lambda: [10, 50])
    # write_to_file_header: Dict[str, Any] = field(default_factory=dict)
    index_examples: List[IndexExample] = field(default_factory=list)

    def write_to_file(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dict_to_write = dict()
        for q_example in self.query_examples:
            dict_to_write[q_example.qid] = q_example.retrieved_iids[:50]
        output_file = os.path.join(output_dir, f"{self.name}_results.json")
        with open(output_file, "w") as f:
            json.dump(dict_to_write, f, indent=4)
        print("Results are written to file", output_file)


def process_img_with_jax(image_path: str, size: int) -> np.ndarray:
    """Process a single image to 224x224 and normalize."""
    img = Image.open(image_path).convert("RGB")
    ima = jnp.array(img)[jnp.newaxis, ...]  # [1, 224, 224, 3]
    ima = ima / (ima.max() + 1e-12)  # avoid 0 division
    ima = jax.image.resize(ima, (1, size, size, 3), method='bilinear')
    return np.array(ima)


def process_img_to_torch(image_path: str, size: int, preprocess=None) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(img)
    else:
        return preprocess(img, return_tensors="pt", input_data_format="channels_last")


class EselTreeDatasetForMagicLens(Dataset):
    def __init__(self, dataset_name: str, tokenizer: Any):

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        index_image_folder = "./data/eseltree/images"  # todo
        index_image_files = list(Path(index_image_folder).glob("*.jpg"))  # todo
        index_image_ids = [file.stem for file in index_image_files]

        query_image_folder = "./data/test/images"  # todo
        query_image_files = list(Path(query_image_folder).glob("*.jpg"))  # todo
        query_image_ids = [file.stem for file in query_image_files]

        null_tokens = tokenizer("")  # used for index example
        null_tokens = np.array(null_tokens)

        self.index_image_folder = index_image_folder
        self.index_image_files = index_image_files
        self.index_image_ids = index_image_ids
        self.query_image_folder = query_image_folder
        self.query_image_files = query_image_files
        self.query_image_ids = query_image_ids
        self.null_tokens = null_tokens

    def prepare_index_examples(self, index_image_ids) -> List[IndexExample]:
        index_examples = []
        with ThreadPoolExecutor() as executor:
            index_example_futures = {executor.submit(self._process_index_example, index_img_id): index_img_id for
                                     index_img_id in index_image_ids}

            with tqdm(total=len(index_image_ids), desc="Index examples") as progress:
                for future in as_completed(index_example_futures):
                    index_example = future.result()
                    index_examples.append(index_example)
                    progress.update(1)
        return index_examples

    def prepare_query_examples(self, query_image_ids) -> List[QueryExample]:
        query_examples = []
        with ThreadPoolExecutor() as executor:
            query_futures = {executor.submit(self._process_query_example, query_img_id): query_img_id for
                             query_img_id in query_image_ids}

            with tqdm(total=len(query_image_ids), desc="Query examples") as progress:
                for future in as_completed(query_futures):
                    q_example = future.result()
                    query_examples.append(q_example)
                    progress.update(1)
        return query_examples

    def _process_index_example(self, index_img_id):
        img_path = os.path.join(self.index_image_folder, index_img_id + ".jpg")
        ima = process_img_with_jax(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=self.null_tokens)

    def _process_query_example(self, query_img_id):
        qtext = ""
        qimage_path = os.path.join(self.query_image_folder, query_img_id + ".jpg")
        ima = process_img_with_jax(qimage_path, 224)
        qtokens = np.array(self.tokenizer(qtext))
        return QueryExample(qid=query_img_id, qtokens=qtokens, qimage=ima, target_iid=0, retrieved_iids=[],
                            retrieved_scores=[])


class EselTreeDatasetDefault(Dataset):
    def __init__(self, dataset_name: str, tokenizer: Any, preprocess=None):

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        index_image_folder = "./data/eseltree/images"  # todo
        index_image_files = list(Path(index_image_folder).glob("*.jpg"))  # todo
        index_image_ids = [file.stem for file in index_image_files]

        query_image_folder = "./data/test/images"  # todo
        query_image_files = list(Path(query_image_folder).glob("*.jpg"))  # todo
        query_image_ids = [file.stem for file in query_image_files]

        null_tokens = tokenizer("")  # used for index example
        null_tokens = np.array(null_tokens)

        self.index_image_folder = index_image_folder
        self.index_image_files = index_image_files
        self.index_image_ids = index_image_ids
        self.query_image_folder = query_image_folder
        self.query_image_files = query_image_files
        self.query_image_ids = query_image_ids
        self.null_tokens = null_tokens
        self.preprocess = preprocess

    def prepare_index_examples(self, index_image_ids) -> List[IndexExample]:
        index_examples = []
        with ThreadPoolExecutor() as executor:
            index_example_futures = {executor.submit(self._process_index_example, index_img_id, self.preprocess):
                                         index_img_id for index_img_id in index_image_ids}

            with tqdm(total=len(index_image_ids), desc="Index examples") as progress:
                for future in as_completed(index_example_futures):
                    index_example = future.result()
                    index_examples.append(index_example)
                    progress.update(1)
        return index_examples

    def prepare_query_examples(self, query_image_ids) -> List[QueryExample]:
        query_examples = []
        with ThreadPoolExecutor() as executor:
            query_futures = {executor.submit(self._process_query_example, query_img_id, self.preprocess): query_img_id
                             for
                             query_img_id in query_image_ids}

            with tqdm(total=len(query_image_ids), desc="Query examples") as progress:
                for future in as_completed(query_futures):
                    q_example = future.result()
                    query_examples.append(q_example)
                    progress.update(1)
        return query_examples

    def _process_index_example(self, index_img_id, preprocess=None):
        img_path = os.path.join(self.index_image_folder, index_img_id + ".jpg")
        ima = process_img_to_torch(img_path, 224, preprocess)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=self.null_tokens)

    def _process_query_example(self, query_img_id, preprocess=None):
        qtext = ""
        qimage_path = os.path.join(self.query_image_folder, query_img_id + ".jpg")
        ima = process_img_to_torch(qimage_path, 224, preprocess)
        qtokens = np.array(self.tokenizer(qtext))
        return QueryExample(qid=query_img_id, qtokens=qtokens, qimage=ima, target_iid=0, retrieved_iids=[],
                            retrieved_scores=[])
