from tqdm import tqdm
import numpy as np
import os
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import pymysql
import requests
from io import BytesIO
from typing import List, Dict, Any
from aisum_database import connect_db

# p_key 목록을 읽고,이미지 파일 경로 목록을 생성
class IndexDataset:
    # 경로 설정
    def __init__(self, task_filepath: str, config: Dict[str, Any]):
        self.image_base_path = Path(config["data"]["image_base_path"])
        self.db_config = config["database"]["mysql"]
        self.config = config
        
        self.image_data: List[Dict[str, Any]] = []
        self.n_of_broken_images = 0 
        
        self._load_data(task_filepath)

    #텍스트 파일에서 p_key 목록 읽기
    def _load_p_keys(self, filepath: str) -> List[str]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"ERROR: 파일을 찾을 수 없습니다: {filepath}")
            return []


    # DB에서 p_key 목록에 해당하는 이미지 정보 조회
    def _fetch_data_from_db(self, p_keys: List[str]) -> List[Dict[str, Any]]:
        db_connection = None
        try:
            db_connection = connect_db(self.config)
            
            with db_connection.cursor(pymysql.cursors.DictCursor) as cursor:
                placeholders = ', '.join(['%s'] * len(p_keys))

                db_name = self.db_config.get('db') or self.db_config.get('database')
                table_name = self.db_config.get('table', 'product_list')

                if not table_name:
                    raise ValueError("config.json의 mysql 설정에 'table'이 지정되지 않았습니다.")

                sql = (
                    "SELECT DISTINCT p_key, save_path, save_name, status, img_dn, img_url "
                    f"FROM {db_name}.{table_name} WHERE p_key IN ({placeholders});"
                )
                
                cursor.execute(sql, p_keys)
                return cursor.fetchall()
        
        except (pymysql.Error, ValueError) as e:
            print(f"ERROR: 데이터베이스 처리 중 에러 발생: {e}")
            return []
        finally:
            if db_connection:
                db_connection.close()

    # db 조회 결과 처리
    def _process_results(self, results: List[Dict[str, Any]]):
        for row in results:
            if row.get('save_path') and row.get('save_name'):
                full_path = self.image_base_path / row['save_path'] / row['save_name']
                row['full_path'] = full_path
            else:
                row['full_path'] = None
            self.image_data.append(row)

    
    def _load_data(self, task_filepath: str):
        p_keys = self._load_p_keys(task_filepath)
        if not p_keys:
            print(f"INFO: {task_filepath}에 처리할 작업이 없습니다.")
            return

        db_results = self._fetch_data_from_db(p_keys)
        if db_results:
            self._process_results(db_results)

    # 이미지 파일 경로 리스트
    @property
    def index_image_files(self) -> List[Path]:
        return [item['full_path'] for item in self.image_data]

    # p_key 리스트
    @property
    def index_image_ids(self) -> List[str]:
        return [item['p_key'] for item in self.image_data]
    
    # 데이터셋 크기
    def __len__(self) -> int:
        return len(self.image_data)
    
    # status와 img_dn에 따라 이미지 데이터 필터링            
    def filter_by_status(self, required_status: int, required_img_dn: str):
        self.image_data = [
            item for item in self.image_data
            if item['status'] is not None
            and int(item['status']) == int(required_status)
            and str(item['img_dn']).strip().lower() == str(required_img_dn).strip().lower()
        ]

        #self.index_image_files = [item['full_path'] for item in self.image_data]
        #self.index_image_ids = [item['p_key'] for item in self.image_data]  
        

    def prepare_index_images(self, batch_idx, batch_size):
        batch_files = self.index_image_files[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_ids = self.index_image_ids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_ids = np.array(batch_ids)
        batch_images = []

        for file_path, img_id in zip(batch_files, batch_ids):
            try:
                img = Image.open(file_path).convert("RGB")
                batch_images.append(img)
            except (FileNotFoundError, UnidentifiedImageError, OSError, SyntaxError) as e:
                print(f"⚠️ WARNING: Failed to open image file. Path: {file_path}, Error: {e}")
                self.n_of_broken_images += 1
        return batch_images, batch_ids


#    def truncate_index_images(self, indexed_codes):
#
#        initial_count = len(self.index_image_ids)
#        self.image_data = [item for item in self.image_data if item['p_key'] not in indexed_codes]
#        self.index_image_files = [item['full_path'] for item in self.image_data]
#        self.index_image_ids = [item['p_key'] for item in self.image_data]
#    
#        final_count = len(self.index_image_ids)
#        print(f"INFO: 이미 인덱싱된 {initial_count - final_count}개의 이미지를 제외합니다. (처리 대상: {final_count}개)")    


class QueryDataset:
    def __init__(self, dataset_name, config):
        self.dataset_name = dataset_name
        if dataset_name == "server":
            query_image_folder = Path(config["data"]["query_image_folder_path"]["server"])
        elif dataset_name == "aisum":
            query_image_folder = Path(config["data"]["query_image_folder_path"]["aisum"])
        else:
            raise ValueError("Unknown query dataset name.")
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