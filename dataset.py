from tqdm import tqdm
import numpy as np
import os
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import pymysql
import requests
from io import BytesIO

class IndexDataset:
    def __init__(self, task_filepath, config):

        self.n_of_broken_images = 0
        self.image_data = [] 
        db_connection = None

        try:
            image_base_path = config["data"]["image_base_path"]
            
            with open(task_filepath, 'r', encoding='utf-8') as f:
                p_keys_to_process = [line.strip() for line in f if line.strip()]

            if not p_keys_to_process:
                print(f"INFO: {task_filepath}에 처리할 작업이 없습니다.")
                self.index_image_files = []
                self.index_image_ids = []
                return

            mysql_cfg = config["database"]["mysql"].copy()
            mysql_cfg['charset'] = 'utf8mb4'
            if 'db' in mysql_cfg:
                mysql_cfg['database'] = mysql_cfg.pop('db')
            db_connection = pymysql.connect(**mysql_cfg)

            with db_connection.cursor(pymysql.cursors.DictCursor) as cursor:
                placeholders = ','.join(['%s'] * len(p_keys_to_process))
                sql = f"SELECT DISTINCT p_key, save_path, save_name, status, img_dn, img_url FROM viscuit.crawling_list WHERE p_key IN ({placeholders});"
                      
                
                cursor.execute(sql, p_keys_to_process)
                results = cursor.fetchall()

                for row in results:
                    if row['save_path'] and row['save_name']:
                        full_path = os.path.join(image_base_path, row['save_path'], row['save_name'])
                        row['full_path'] = Path(full_path)
                    else:
                        row['full_path'] = None
                    self.image_data.append(row)  

            self.index_image_files = [item['full_path'] for item in self.image_data]
            self.index_image_ids = [item['p_key'] for item in self.image_data]
            
        except Exception as e:
            print(f"ERROR: 데이터셋 생성 중 에러 발생: {e}")
        finally:
            if db_connection:
                db_connection.close()
                
    def filter_by_status(self, required_status: int, required_img_dn: str):
        """
        주어진 status와 img_dn 조건에 맞게 데이터셋을 필터링합니다.
        """
        self.image_data = [
            item for item in self.image_data
            if item['status'] is not None
            and int(item['status']) == int(required_status)
            and str(item['img_dn']).strip().lower() == str(required_img_dn).strip().lower()
        ]
        # 필터링된 결과를 기반으로 파일/ID 목록 다시 생성
        self.index_image_files = [item['full_path'] for item in self.image_data]
        self.index_image_ids = [item['p_key'] for item in self.image_data]  
        

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


    def truncate_index_images(self, indexed_codes):

        initial_count = len(self.index_image_ids)
        
        self.image_data = [item for item in self.image_data if item['p_key'] not in indexed_codes]
        
        self.index_image_files = [item['full_path'] for item in self.image_data]
        self.index_image_ids = [item['p_key'] for item in self.image_data]
        
        final_count = len(self.index_image_ids)
        print(f"INFO: 이미 인덱싱된 {initial_count - final_count}개의 이미지를 제외합니다. (처리 대상: {final_count}개)")    


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