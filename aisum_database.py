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
        local_db = self.repository.databases.get_db_by_name(self.model_name)
        dataset = QueryDataset("aisum", self.config)
        detection_model = YOLO("yolo", self.config)
        batch_size = self.config["model"].get(self.model_name, {}).get("batch_size", 4)
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

            if self.model_name != "ensemble":
                retrieval_result = self.repository.get_retrieval_result_by_name(self.model_name, detection_images,
                                                                                detection_ids,
                                                                                detection_classes)
            else:
                retrieval_result = self.repository.ensemble(detection_images, detection_ids, detection_classes)
            self.repository.clear_retrieval_results()

            for result_ids, p_scores, segment_id in zip(retrieval_result["result_ids"], retrieval_result["p_scores"],
                                                        detection_segment_ids):
                local_db.insert_search_results(segment_id, result_ids, p_scores)

        # 처리된 id 범위 출력
        id_range = local_db.get_search_results_id_range()
        if id_range:
            print(f"[INFO] 모든 결과 저장 완료 (id: {id_range[0]} - {id_range[1]})")
        else:
            print("[INFO] search_results 테이블에 데이터가 없습니다.")

    def fill_missing_columns_from_aisum_db(self):
        local_db = self.repository.databases.get_db_by_name(self.model_name)
        rows = local_db.get_rows_with_missing_columns()
        print(f"[INFO] 채워야 할 row 개수: {len(rows)}")

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
            print(f"[INFO] search_results 컬럼 채우기 완료 (id: {id_range[0]} - {id_range[1]})")
        else:
            print("[INFO] search_results 테이블에 데이터가 없습니다.")

    def copy_to_mysql_db(self):
        local_db = self.repository.databases.get_db_by_name(self.model_name)

        # Fetch only top30 results for the specified model_name
        rows = local_db.get_search_results_30(self.model_name)
        print(f"[INFO] PostgreSQL에서 {len(rows) if rows else 0}개 row(top30) 조회 완료")

        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()

        try:
            now = datetime.now()
            ymdh = int(now.strftime('%Y%m%d%H'))
        
            for row in tqdm(rows or [], desc="MySQL Insert (top30)"):
                model_name, pu_id, place_id, c_key, au_id, p_key, p_category, p_score = row

                query = """
                    INSERT INTO pm_test_2nd_content_list 
                    (ymdh, model_name, pu_id, place_id, c_key, au_id, p_key, p_category, p_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    float(p_score) if p_score is not None else None
                ))

            conn.commit()

            print(f"[INFO] MySQL에 {len(rows) if rows else 0}개 row(top30) 저장 완료")

        except Exception as e:
            print(f"[ERROR] MySQL 저장 중 오류 발생: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def save_top30_per_query_id(self, model_name=None):
        local_db = self.repository.databases.get_db_by_name(self.model_name)
        local_db.save_top30_per_query_id(model_name)


if __name__ == "__main__":
    print("==== 메뉴 ====")
    print("1. 검색결과 DB 저장 및 컬럼 채우기")
    print("2. top30 추출하여 저장")
    print("3. AISUM DB에 데이터 저장")

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    menu = input("작업 번호를 선택하세요 (1/2/3): ").strip()
    model_input = input("Image Embedding Model Name (단일 모델명 또는 'ensemble'): ").strip()
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
        db_adapter.copy_to_mysql_db()
    else:
        print("[ERROR] 올바른 메뉴 번호를 입력하세요.")
