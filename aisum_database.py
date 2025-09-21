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

def get_p_key_from_result_id(result_id):
    """result_id에서 p_key 추출"""
    parts = str(result_id).split('_')
    if len(parts) > 1:
        return '_'.join(parts[:-1])
    else:
        return str(result_id)

def deduplicate_results_per_object(result_ids, similarities, p_scores):
    """객체별로 p_key 기준 중복 제거"""
    seen_p_keys = {}
    
    for idx, (result_id, similarity, p_score) in enumerate(zip(result_ids, similarities, p_scores)):
        p_key = get_p_key_from_result_id(result_id)
        
        # 빈 p_key 필터링
        if not p_key or p_key.strip() == '':
            continue
            
        if p_key not in seen_p_keys or p_score > seen_p_keys[p_key]['p_score']:
            seen_p_keys[p_key] = {
                'result_id': result_id,
                'similarity': similarity,
                'p_score': p_score,
                'idx': idx
            }
    
    # 중복 제거된 결과 반환
    dedup_result_ids = []
    dedup_similarities = []
    dedup_p_scores = []
    
    for p_key, data in seen_p_keys.items():
        dedup_result_ids.append(data['result_id'])
        dedup_similarities.append(data['similarity'])
        dedup_p_scores.append(data['p_score'])
    
    return dedup_result_ids, dedup_similarities, dedup_p_scores

def get_image_urls_from_mysql(config, p_keys):
    # p_key를 이용해 상품 이미지 URL을 MySQL(pm_test_product_list)에서 조회
    conn = connect_db(config)
    cur = conn.cursor()
    
    try:
        if not p_keys:
            return []
            
        placeholders = ','.join(['%s'] * len(p_keys))
        query = f"""
            SELECT p_key, img_url 
            FROM piclick.pm_test_product_list 
            WHERE p_key IN ({placeholders})
        """
        cur.execute(query, p_keys)
        results = cur.fetchall()
        
        # p_key -> img_url 매핑 딕셔너리 생성
        url_map = {row[0]: row[1] for row in results}
        
        # 원래 순서대로 URL 리스트 반환
        urls = [url_map.get(p_key, '') for p_key in p_keys]
        return urls
        
    except Exception as e:
        print(f"Error fetching image URLs from MySQL: {e}")
        return [''] * len(p_keys)
    finally:
        cur.close()
        conn.close()

class AisumDBAdapter:
    def __init__(self, model_name, config):
        repository = ImageRetrievalRepository(config)

        self.repository = repository
        self.model_name = model_name
        self.config = config
        self.k_per_object = config["retrieval"]["k_per_object"]
        self.retrieval_mode = config["retrieval"]["category_filter"]

    def save_search_results_to_mysql(self):
        dataset = QueryDataset("aisum", self.config)
        detection_model = YOLO("yolo", self.config)
        batch_size = self.config["model"].get(self.model_name, {}).get("batch_size", 8)
        total_images = len(dataset.query_image_files)
        total_batches = total_images // batch_size + (1 if total_images % batch_size > 0 else 0)

        conn = connect_db(self.config)
        cur = conn.cursor()

        #total_inserts = 0 ##
        try:
            for batch_idx in tqdm(range(total_batches), desc=f"Processing {total_images} Images(batch size={batch_size})"):
                test_images, test_ids = dataset.prepare_query_images(batch_idx, batch_size)
                if not test_images:
                    continue
                
                # YOLO 탐지
                detection_result = detection_model(test_images, test_ids)
                detection_images = detection_result["detection_images"]
                detection_ids = detection_result["original_image_ids"]
                detection_segment_ids = detection_result["image_segment_ids"]
                detection_classes = detection_result["detection_classes"]
                detection_coordinates = detection_result["detection_coordinates"]  # (x1,y1,x2,y2)
                detection_sizes = detection_result["detection_sizes"]
                detection_centrality = detection_result["detection_centrality"]
                detection_confidences = detection_result["detection_confidences"]  # YOLO confidence

                #print(f"DEBUG: YOLO detection - Found {len(detection_images)} objects")
                #print(f"DEBUG: Segment IDs: {detection_segment_ids[:3]}...")  # 처음 3개만
                #print(f"DEBUG: Classes: {detection_classes[:3]}...")
                
                # 검색 수행
                if self.model_name != "ensemble":
                    retrieval_result = self.repository.get_retrieval_result_by_name(
                        self.model_name, 
                        detection_images,
                        detection_segment_ids,
                        detection_classes,
                        detection_sizes,
                        detection_centrality
                    )
                else:
                    retrieval_result = self.repository.ensemble(
                        detection_images, 
                        detection_segment_ids,
                        detection_classes,
                        detection_sizes,
                        detection_centrality
                    )
                '''
                # 디버깅 출력 
                print(f"DEBUG: Retrieval result keys: {retrieval_result.keys()}")
                print(f"DEBUG: Result IDs length: {len(retrieval_result['result_ids'])}")
                
                batch_insert_count = 0
                for i, result_ids in enumerate(retrieval_result["result_ids"]):
                    batch_insert_count += len(result_ids)
                    print(f"DEBUG: Object {i}: {len(result_ids)} results")
                    
                print(f"DEBUG: Total items to insert in this batch: {batch_insert_count}")
                
                if batch_insert_count == 0:
                    print("WARNING: No search results found! Skipping insert.")
                    continue
                # 디버깅 출력 끝
                '''
                self.repository.clear_retrieval_results()
                # MySQL에 검색 결과 저장
                for result_ids, similarities, p_scores, segment_id, category, bbox, bbox_size, bbox_centrality, yolo_conf in zip(
                    retrieval_result["result_ids"],
                    retrieval_result["similarities"],
                    retrieval_result.get("p_scores", retrieval_result.get("p_score", retrieval_result["similarities"])),
                    detection_segment_ids, detection_classes, detection_coordinates, 
                    detection_sizes, detection_centrality, detection_confidences):
                    
                    self.insert_search_results_into_mysql(cur, segment_id, result_ids, similarities, p_scores, 
                                                    category, bbox, bbox_size, bbox_centrality, yolo_conf)
                    '''
                    if len(result_ids) > 0:  # 결과가 있을 때만 실행
                        print(f"DEBUG: Inserting {len(result_ids)} items for segment {segment_id}")
                        self.insert_search_results_into_mysql(cur, segment_id, result_ids, similarities, p_scores, 
                                                            category, bbox, bbox_size, bbox_centrality, yolo_conf)
                        total_inserts += len(result_ids)
                    else:
                        print(f"DEBUG: No results for segment {segment_id}")
                    '''

            #print(f"DEBUG: Total items inserted: {total_inserts}") ##
            conn.commit()
            # 실제 삽입 확인
            #cur.execute("SELECT COUNT(*) FROM viscuit.2nd_content_list WHERE ymdh = %s", (int(datetime.now().strftime("%Y%m%d%H")),))
            #actual_count = cur.fetchone()[0]
            #print(f"DEBUG: Actual rows in database: {actual_count}")
            #---------------------
            self.fill_missing_columns()

        except Exception as e:
            print(f"Error save search_result into MySQL: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def insert_search_results_into_mysql(self, cur, segment_id, result_ids, similarities, p_scores, category, bbox, bbox_size, bbox_centrality, yolo_conf):
        table_name = "viscuit.2nd_content_list"
        yolo_version = self.config["model"]["yolo"]["version"]
        embedding_model_name = self.model_name if self.model_name != "ensemble" else "ensemble"
        detection_model_name = f"yolo{yolo_version}"

        def get_p_category(segment_id):
            try:
                num = str(segment_id).split('_')[-1]
                if num.isdigit():
                    return f"Object-{num}"
                else:
                    return f"Object-0"
            except Exception:
                    return "Object-0"
        def get_puid_ckey(segment_id):
            parts = str(segment_id).split('_')
            if len(parts) >= 2:
                pu_id = parts[0]
                c_key = parts[1]
                return pu_id, c_key
            else:
                return None, None
        def get_retrieval_mode(retrieval_mode, category):
            if retrieval_mode == "category_retrieval":
                return f"{retrieval_mode}_{category}"
            else:
                return retrieval_mode

        # 쿼리 객체 1개의 검색 결과 중 p_key 기준으로 중복 제거 (p_score가 높은 것만 유지)
        dedup_result_ids, dedup_similarities, dedup_p_scores = deduplicate_results_per_object(
            result_ids, similarities, p_scores)

        p_category = get_p_category(segment_id)
        pu_id, c_key = get_puid_ckey(segment_id)
        x1, y1, x2, y2 = bbox if isinstance(bbox, tuple) else (0, 0, 0, 0)
        retrieval_mode_value = get_retrieval_mode(self.retrieval_mode, category)
        insert_count = 0 ##
        try: 
            for result_id, similarity, p_score in zip(dedup_result_ids, dedup_similarities, dedup_p_scores):
                p_key = get_p_key_from_result_id(result_id)
                now = datetime.now()
                ymdh = int(now.strftime("%Y%m%d%H"))
                
                query = f"""
                    INSERT INTO {table_name}
                    (ymdh, pu_id, c_key, p_key, p_category, p_score,
                    embedding_model_name, detection_model_name,
                    category, conf, x1, y1, x2, y2, retrieval_mode)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                try:
                    cur.execute(query, (
                        ymdh,
                        pu_id,
                        c_key,
                        p_key,
                        p_category,
                        float(p_score) if p_score is not None else None,
                        embedding_model_name,
                        detection_model_name,
                        category,
                        float(yolo_conf) if yolo_conf is not None else None,
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        retrieval_mode_value
                    ))
                    insert_count += 1
                
                except Exception as e:
                    print(f"Error inserting individual result: {e}")
                    raise
            
            #print(f"DEBUG: Successfully inserted {insert_count} rows for {segment_id}") ##
        except Exception as e:
            print(f"Error inserting search results into MySQL: {e}")
            #-------------
            #print(f"DEBUG: Failed on segment {segment_id}")
            #import traceback
            #traceback.print_exc()
            #-------------
            raise
    
    def fill_missing_columns(self):
        """MySQL에서 누락된 컬럼들을 조회 후 채우기"""
        conn = connect_db(self.config)
        cur = conn.cursor()
        
        try:
            # 누락된 컬럼이 있는 행들 조회
            rows = self.get_rows_with_missing_columns(cur)
            print(f"[INFO] missing rows: {len(rows)}")
            
            # 배치로 메타데이터 조회 (성능 최적화)
            content_map, product_map = self.batch_fetch_metadata(cur, rows)
            
            # 배치 업데이트 준비
            update_data = []
            
            for row_id, pu_id, c_key, p_key in tqdm(rows, desc="Preparing missing columns data"):
                # content_list에서 조회
                content = content_map.get((pu_id, c_key)) if pu_id and c_key else None
                p_set_id, slot_id = content if content else (None, None)
                
                # product_list에서 조회
                au_id = product_map.get(p_key)
                
                update_data.append((p_set_id, slot_id, au_id, row_id))
            
            # 배치 업데이트 실행
            if update_data:
                self.batch_update_missing_columns(cur, update_data)
                conn.commit()
                print(f"[INFO] Finished filling missing columns for {len(update_data)} rows")
            
        except Exception as e:
            print(f"Error filling missing columns: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def get_rows_with_missing_columns(self, cur):
        """누락된 컬럼이 있는 행들을 조회하는 함수"""
        table_name = "viscuit.2nd_content_list"
        try:
            query = f"""
                SELECT id, p_key, pu_id, c_key
                FROM {table_name}
                WHERE (p_set_id IS NULL OR p_set_id = '') 
                    OR (pu_id IS NULL OR pu_id = 0) 
                    OR (slot_id IS NULL OR slot_id = 0) 
                    OR (au_id IS NULL OR au_id = 0)
            """
            cur.execute(query)
            rows = cur.fetchall()  # [(id, p_key), ...]
            
            enhanced_rows = []
            for row_id, p_key, pu_id, c_key in rows:
                enhanced_rows.append((row_id, pu_id, c_key, p_key))
            return enhanced_rows
                
        except Exception as e:
            print(f"Error fetching rows with missing columns: {e}")
            return []

    def batch_fetch_metadata(self, cur, rows):
        # 고유한 pu_id, c_key, p_key 추출
        unique_puids_ckeys = list(set([(row[1], row[2]) for row in rows if row[1] and row[2]]))
        unique_p_keys = list(set([row[3] for row in rows if row[3]]))
        
        # content_list 배치 조회
        content_map = {}
        if unique_puids_ckeys:
            for pu_id, c_key in unique_puids_ckeys:
                cur.execute("""
                            SELECT p_set_id, place_id FROM piclick.pm_test_content_list
                            WHERE pu_id = %s AND c_key = %s
                            """, (pu_id, c_key))
                result = cur.fetchone()
                if result:
                    content_map[(pu_id, c_key)] = result  #(p_set_id, place_id)
        
        # product_list 배치 조회
        product_map = {}
        if unique_p_keys:
            placeholders = ','.join(['%s'] * len(unique_p_keys))
            cur.execute(f"""
                SELECT p_key, au_id FROM piclick.pm_test_product_list WHERE p_key IN ({placeholders})
            """, unique_p_keys)
            for row in cur.fetchall():
                product_map[row[0]] = row[1]
        
        return content_map, product_map

    def batch_update_missing_columns(self, cur, update_data):
        """배치로 누락된 컬럼들을 업데이트하는 함수"""
        table_name = "viscuit.2nd_content_list"
        try:
            query = f"""
                UPDATE {table_name}
                SET p_set_id = %s, slot_id = %s, au_id = %s
                WHERE id = %s
            """
            cur.executemany(query, update_data)
        except Exception as e:
            print(f"Error batch updating missing columns: {e}")
            raise
    
    @staticmethod
    def get_image_urls_for_server(config, result_ids):
        # 쿼리이미지 server에서 사용할 이미지URL 조회
        retrieved_image_urls = []
        for result_id_list in result_ids:
            # p_key 추출
            p_keys = []
            for img_id in result_id_list:
                p_key = get_p_key_from_result_id(img_id)
                p_keys.append(p_key)
            
            # MySQL에서 이미지 URL 조회
            img_urls = get_image_urls_from_mysql(config, p_keys)
            retrieved_image_urls.append(img_urls)
        return retrieved_image_urls
    
    # --------------------

    def save_search_results_to_local_db(self):
        #순환 참조 일으켜서 필요한 부분에서 import
        from dataset import QueryDataset
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
            detection_coordinates = detection_result["detection_coordinates"] # search_result table의 bbox로 들어감
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

    def copy_to_mysql_db(self):
        local_db = self.repository.databases.get_db_by_name(self.model_name)

        # Fetch only top30 results for the specified model_name
        rows = local_db.get_search_results_30(self.model_name)
        print(f"[INFO] Retrieved {len(rows) if rows else 0} rows (top30) from PostgreSQL")

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


if __name__ == "__main__":
    print("========= MENU =========")
    print("1. Save search results to DB and fill columns")
    print("2. Extract and save top30")
    print("3. Save data to AISUM DB")
    print("[New] 4. Save search results to AISUM DB")  # viscuit.2nd_content_list에 검색결과 직접 insert
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
        db_adapter.copy_to_mysql_db()
    elif menu == "4":
        # 검색결과를 mysql로 직접 insert -> 참고테이블 사용해서 공백 컬럼 채움 
        db_adapter.save_search_results_to_mysql()
    else:
        print("[ERROR] Please enter a valid menu number.")
