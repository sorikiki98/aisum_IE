import json
from datetime import datetime
import pymysql
from tqdm import tqdm

from dataset import QueryDataset
from repository import ImageRetrievalRepository
from yolo import YOLO


def connect_db(config):
    try:
        return pymysql.connect(**config["database"]["mysql"])
    except Exception as e:
        raise ValueError(f"Error connecting to database: {e}")


class AisumDBAdapter:
    def __init__(self, model_name, config):
        repository = ImageRetrievalRepository(config)

        self.repository = repository
        self.model_name = model_name
        self.config = config

    def save_search_results_to_local_db(self):
        dataset = QueryDataset("aisum", self.config)
        detection_model = YOLO("yolo", self.config)
        local_db = self.repository.databases.get_db_by_name(self.model_name)
        batch_size = self.config["model"].get(self.model_name, {}).get("batch_size", 8)
        total_images = len(dataset.query_image_files)
        total_batches = total_images // batch_size + (1 if total_images % batch_size > 0 else 0)

        local_db.create_search_results_table()
        for batch_idx in tqdm(range(total_batches), desc=f"Processing {total_images} Images (batch size={batch_size})"):
            test_images, test_ids = dataset.prepare_query_images(batch_idx, batch_size)
            if not test_images:
                continue
            detection_result = detection_model(test_images, test_ids)
            detection_images = detection_result["detection_images"]
            detection_ids = detection_result["original_image_ids"]
            detection_segment_ids = detection_result["image_segment_ids"]
            detection_classes = detection_result["detection_classes"]
            detection_coordinates = detection_result["detection_coordinates"]
            detection_sizes = detection_result["detection_sizes"]
            detection_centrality = detection_result["detection_centrality"]

            if self.model_name != "ensemble":
                retrieval_result = self.repository.get_retrieval_result_by_name(self.model_name, detection_images,
                                                                                detection_ids,
                                                                                detection_classes)
            else:
                retrieval_result = self.repository.ensemble(detection_images, detection_ids, detection_classes)
            self.repository.clear_retrieval_results()

            for result_ids, similarity, segment_id, category, bbox, bbox_size, bbox_centrality in zip(
                    retrieval_result["result_ids"],
                    retrieval_result["similarities"],
                    detection_segment_ids, detection_classes,
                    detection_coordinates, detection_sizes, detection_centrality):
                local_db.insert_search_results(segment_id, result_ids, similarity, category, bbox, bbox_size,
                                               bbox_centrality)

        # 처리된 id 범위 출력
        id_range = local_db.get_search_results_id_range()
        if id_range:
            print(f"[INFO] All results saved (id: {id_range[0]} - {id_range[1]})")
        else:
            print("[INFO] No data in search_results table.")

        # p_score 컬럼 업데이트

        local_db.update_p_score_column()

    def fill_missing_columns_from_aisum_db(self):
        local_db = self.repository.databases.get_db_by_name(self.model_name)
        rows = local_db.get_rows_with_missing_columns()
        print(f"[INFO] missing rows: {len(rows)}")

        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        for id, query_id, p_key in tqdm(rows, desc="Filling missing columns"):
            # content_list
            cur.execute("SELECT pu_id, place_id, c_key FROM pm_test_content_list WHERE id=%s", (query_id,))
            content = cur.fetchone()
            pu_id, place_id, c_key = content if content else (None, None, None)
            # product_list
            cur.execute("SELECT au_id FROM pm_test_product_list WHERE p_key=%s", (p_key,))
            product = cur.fetchone()
            au_id = product[0] if product else None
            # PostgreSQL에 업데이트
            local_db.update_search_results_columns(id, pu_id, place_id, c_key, au_id)
        cur.close()
        conn.close()

        # 처리된 id 범위 출력
        id_range = local_db.get_search_results_id_range()
        if id_range:
            print(f"[INFO] Finished filling search_results columns (id: {id_range[0]} - {id_range[1]})")
        else:
            print("[INFO] No data in search_results table.")

    def copy_to_mysql_db_top30(self):
        local_db = self.repository.databases.get_db_by_name(self.model_name)

        rows = local_db.get_search_results_30(self.model_name)
        #rows = local_db.get_search_results(self.model_name)
        print(f"[INFO] Retrieved {len(rows) if rows else 0} rows from PostgreSQL")

        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()

        try:
            now = datetime.now()
            ymdh = int(now.strftime('%Y%m%d%H'))

            for row in tqdm(rows or [], desc="MySQL Insert (top30)"):
                model_name, pu_id, place_id, c_key, au_id, p_key, p_category, p_score, category, bbox = row

                query = """
                    INSERT INTO pm_test_2nd_content_list 
                    (ymdh, model_name, pu_id, place_id, c_key, au_id, p_key, p_category, p_score, category, box)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cur.execute(query, (
                    ymdh,
                    model_name,
                    pu_id,
                    place_id,
                    c_key,
                    au_id,
                    p_key,
                    p_category,
                    float(p_score) if p_score is not None else None,
                    category,
                    bbox
                ))

            conn.commit()

            print(f"[INFO] Saved {len(rows) if rows else 0} rows (top30) to MySQL")

        except Exception as e:
            print(f"[ERROR] Error occurred while saving to MySQL: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def save_top30_per_query_id(self, model_name=None):
        local_db = self.repository.databases.get_db_by_name(self.model_name)
        local_db.save_top30_per_query_id(model_name)

    def copy_to_mysql_db_all(self):
        local_db = self.repository.databases.get_db_by_name(self.model_name)

        # 중복 제거
        local_db.remove_duplicates_from_search_results()

        rows = local_db.get_search_results(self.model_name)
        print(f"[INFO] Retrieved {len(rows) if rows else 0} rows from PostgreSQL")

        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()

        try:
            now = datetime.now()
            ymdh = int(now.strftime('%Y%m%d%H'))

            for row in tqdm(rows or [], desc="MySQL Insert (all)"):
                model_name, pu_id, place_id, c_key, au_id, p_key, p_category, p_score, category, bbox = row

                query = """
                    INSERT INTO pm_test_2nd_content_box_list 
                    (ymdh, model_name, pu_id, place_id, c_key, au_id, p_key, p_category, p_score, category, box)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cur.execute(query, (
                    ymdh,
                    model_name,
                    pu_id,
                    place_id,
                    c_key,
                    au_id,
                    p_key,
                    p_category,
                    float(p_score) if p_score is not None else None,
                    category,
                    bbox
                ))

            conn.commit()

            print(f"[INFO] Saved {len(rows) if rows else 0} rows (top30) to MySQL")

        except Exception as e:
            print(f"[ERROR] Error occurred while saving to MySQL: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()


if __name__ == "__main__":
    print("========= MENU =========")
    print("1. Save search results to DB and fill columns")
    print("2. Extract and save top30")
    print("3. Save data to AISUM DB (top30)")
    print("4. Save data to AISUM DB (all)")
    print("=========================")

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    menu = input("Select task number (1/2/3/4): ").strip()
    model_input = input("Image Embedding Model Name (single model name or 'ensemble'): ").strip()
    if menu == "1":
        if model_input not in config["model"] and model_input != "ensemble":
            raise ValueError("Invalid embedding model name.")

    db_adapter = AisumDBAdapter(model_input, config)

    if menu == "1":
        db_adapter.save_search_results_to_local_db()
        db_adapter.fill_missing_columns_from_aisum_db()
    elif menu == "2":
        local_db = db_adapter.repository.databases.get_db_by_name(model_input)
        local_db.save_top30_per_query_id(model_name=model_input)
    elif menu == "3":
        db_adapter.copy_to_mysql_db_top30()
    elif menu == "4":
        db_adapter.copy_to_mysql_db_all()
    else:
        print("[ERROR] Please enter a valid menu number.")
