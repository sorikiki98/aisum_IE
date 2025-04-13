import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
    # todo: server 연동 시, 아래 코드 사용
    #category1_code: str 
    #category2_code: str


@dataclass
class IndexExample:
    iid: Union[int, str]
    iimage: Union[np.ndarray, torch.Tensor, dict]
    itokens: np.ndarray
    category1_code: str
    category2_code: str


@dataclass
class Dataset:
    name: str
    query_examples: List[QueryExample] = field(default_factory=list)
    k_range: List[int] = field(default_factory=lambda: [10, 50])
    # write_to_file_header: Dict[str, Any] = field(default_factory=dict)
    index_examples: List[IndexExample] = field(default_factory=list)

    '''
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
    '''


def process_img_with_jax(image_path: str, size: int) -> np.ndarray:
    """Process a single image to 224x224 and normalize."""
    img = Image.open(image_path).convert("RGB")
    ima = jnp.array(img)[jnp.newaxis, ...]  # [1, 224, 224, 3]
    ima = ima / (ima.max() + 1e-12)  # avoid 0 division
    ima = jax.image.resize(ima, (1, size, size, 3), method='bilinear')
    return np.array(ima)


def process_img_to_torch(image_path: str, size: int, preprocess=None, prompt=None) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")

    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(img)
    elif "SiglipProcessor" in str(type(preprocess)) or "SAM2" in str(type(preprocess)):  # SigLIP 또는 SAM2 processor인 경우
        return img
    else:
        if prompt is None:
            return preprocess(image_path, return_tensors="pt", input_data_format="channels_last")
            # return processed["pixel_values"]
            # return processed(img)
        else:
            # return preprocess(images=img, return_tensors="pt", text=prompt)
            return preprocess(img)


class EselTreeDatasetForMagicLens(Dataset):
    def __init__(self, dataset_name: str, tokenizer: Any):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

        index_image_folder = "./data/eseltree/images"
        index_image_files = sorted(Path(index_image_folder).glob("**/*.jpg"))
        index_image_ids_with_cats = [str(file).split(".")[0] for file in index_image_files]
        index_image_ids = [file.stem for file in index_image_files]

        query_image_folder = "./data/test"
        query_image_files = sorted(Path(query_image_folder).glob("*.jpg"))
        query_image_ids = [file.stem for file in query_image_files]

        null_tokens = tokenizer("")  # used for index example
        null_tokens = np.array(null_tokens)

        self.index_image_folder = index_image_folder
        self.index_image_files = index_image_files
        self.index_image_ids_with_cats = index_image_ids_with_cats
        self.index_image_ids = index_image_ids
        self.query_image_folder = query_image_folder
        self.query_image_files = query_image_files
        self.query_image_ids = query_image_ids
        self.null_tokens = null_tokens

    def prepare_index_examples(self, index_image_ids) -> List[IndexExample]:
        index_examples = []
        with tqdm(total=len(index_image_ids), desc="Index examples") as progress:
            for index_img_id in index_image_ids:
                index_example = self._process_index_example(index_img_id)
                index_examples.append(index_example)
                progress.update(1)
        return index_examples

    def prepare_query_examples(self, query_image_ids) -> List[QueryExample]:
        query_examples = []
        with tqdm(total=len(query_image_ids), desc="Query examples") as progress:
            for query_img_id in query_image_ids:
                q_example = self._process_query_example(query_img_id)
                query_examples.append(q_example)
                progress.update(1)
        return query_examples

    def _process_index_example(self, index_img_id):  # cat1/cat2/img_id.jpg
        cat1_code = index_img_id.split(os.sep)[-3]
        cat2_code = index_img_id.split(os.sep)[-2]
        img_id = index_img_id.split(os.sep)[-1]
        img_path = os.path.join(self.index_image_folder, cat1_code, cat2_code, img_id + ".jpg")
        ima = process_img_with_jax(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=self.null_tokens, category1_code=cat1_code,
                            category2_code=cat2_code)

    def _process_query_example(self, query_img_id):
        # todo: server 연동 시, 아래 코드 사용
        # cat1_code = query_img_id.split(os.sep)[-3]
        # cat2_code = query_img_id.split(os.sep)[-2]
        # img_id = query_img_id.split(os.sep)[-1]
        qtext = ""
        qimage_path = os.path.join(self.query_image_folder, query_img_id + ".jpg")
        ima = process_img_with_jax(qimage_path, 224)
        qtokens = np.array(self.tokenizer(qtext))
        return QueryExample(qid=query_img_id, qtokens=qtokens, qimage=ima, target_iid=0, retrieved_iids=[],
                            retrieved_scores=[])


class EselTreeDatasetDefault(Dataset):
    def __init__(self, dataset_name: str, tokenizer: Any, preprocess=None, prompt=None):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

        project_root = Path(__file__).parent
        index_image_folder = project_root / "data_three" / "eseltree" / "images"
        index_image_files = sorted(index_image_folder.glob("**/*.jpg"))
        index_image_ids_with_cats = [str(file).split(".")[0] for file in index_image_files]
        index_image_ids = [file.stem for file in index_image_files]

        query_image_folder = project_root / "data" / "test" / "images"
        print(f"Looking for query images in: {query_image_folder}")
        query_image_files = sorted(query_image_folder.glob("*.jpg"))
        query_image_ids = [file.stem for file in query_image_files]

        null_tokens = tokenizer("")  # used for index example
        null_tokens = np.array(null_tokens)

        self.index_image_folder = index_image_folder
        self.index_image_files = index_image_files
        self.index_image_ids = index_image_ids
        self.index_image_ids_with_cats = index_image_ids_with_cats
        self.query_image_folder = query_image_folder
        self.query_image_files = query_image_files
        self.query_image_ids = query_image_ids
        self.null_tokens = null_tokens
        self.preprocess = preprocess
        self.prompt = prompt

    def prepare_index_examples(self, index_image_ids) -> List[IndexExample]:
        index_examples = []
        with tqdm(total=len(index_image_ids), desc="Index examples") as progress:
            for index_img_id in index_image_ids:
                index_example = self._process_index_example(index_img_id, preprocess=self.preprocess,
                                                            prompt=self.prompt)
                index_examples.append(index_example)
                progress.update(1)
        return index_examples

    def prepare_query_examples(self, query_image_ids) -> List[QueryExample]:
        query_examples = []
        with tqdm(total=len(query_image_ids), desc="Query examples") as progress:
            for query_img_id in query_image_ids:
                q_example = self._process_query_example(query_img_id, preprocess=self.preprocess)
                query_examples.append(q_example)
                progress.update(1)
        return query_examples

    def _process_index_example(self, index_img_id, preprocess=None, prompt=None):  # cat1/cat2/img_id.jp
        cat1_code = index_img_id.split(os.sep)[-3]
        cat2_code = index_img_id.split(os.sep)[-2]
        img_id = index_img_id.split(os.sep)[-1]
        img_path = os.path.join(self.index_image_folder, cat1_code, cat2_code, img_id + ".jpg")
        ima = process_img_to_torch(img_path, 224, preprocess, prompt)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=self.null_tokens, category1_code=cat1_code,
                            category2_code=cat2_code)

    def _process_query_example(self, query_img_id, preprocess=None):  # cat1/cat2/img_id.jpg
        qtext = ""
        qimage_path = os.path.join(self.query_image_folder, query_img_id + ".jpg")
        print(qimage_path)
        ima = process_img_to_torch(qimage_path, 224, preprocess)
        qtokens = np.array(self.tokenizer(qtext))
        return QueryExample(qid=query_img_id, qtokens=qtokens, qimage=ima, target_iid=0, retrieved_iids=[],
                            retrieved_scores=[])
