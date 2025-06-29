from datetime import datetime

import psycopg2
import time
from tqdm import tqdm


def connect_db(config):
    try:
        return psycopg2.connect(**config["database"]["postgres"])
    except Exception as e:
        raise ValueError(f"Error connecting to database: {e}")


class PGVectorDB:
    def __init__(self, image_embedding_model_name, config):
        conn = connect_db(config)

        if image_embedding_model_name in config["model"]:
            model_config = config["model"][image_embedding_model_name]
            table_name = f"image_embeddings_{image_embedding_model_name}"
            num_dimensions = model_config["num_dimension"]

            cur = conn.cursor()
            try:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id VARCHAR(255) PRIMARY KEY,
                        code VARCHAR(255),
                        embedding VECTOR({num_dimensions}),
                        category VARCHAR(500)
                    );
                """)
                conn.commit()
            except Exception as e:
                print(f"Error creating table: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        self.config = config
        self.image_embedding_model_name = image_embedding_model_name

    def get_yolo_version(self):
        return self.config["model"]["yolo"]["version"]

    def insert_embeddings(self, ids, img_codes, image_embeddings, cats):
        table_name = f"image_embeddings_{self.image_embedding_model_name}"

        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        try:
            for id, code, embedding, cat in zip(ids, img_codes, image_embeddings, cats):
                query = f"""
                    INSERT INTO {table_name} (id, code, embedding, category)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                """
                embedding = embedding.tolist()
                cur.execute(query, (str(id), str(code), embedding, cat))
            conn.commit()

        except Exception as e:
            print(f"Error inserting into PostgreSQL: {e}")
            conn.rollback()

        finally:
            cur.close()
            conn.close()

    def create_index(self):
        table_name = f"image_embeddings_{self.image_embedding_model_name}"

        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()

        try:
            cur = conn.cursor()
            index_query = f"""
                CREATE INDEX IF NOT EXISTS hnsw_idx_{self.image_embedding_model_name}
                ON "{table_name}" USING hnsw (embedding vector_cosine_ops)
                WITH (m = 32, ef_construction = 300);
            """
            cur.execute(index_query)
            conn.commit()

        except Exception as e:
            print("Error creating HNSW index:")
            print(index_query)
            print("Error:", e)
            conn.rollback()

        finally:
            cur.close()
            conn.close()

    def optimize_index(self):
        table_name = f"image_embeddings_{self.image_embedding_model_name}"
        index_name = f"hnsw_idx_{self.image_embedding_model_name}"
        config = self.config

        conn = connect_db(config)
        cur = conn.cursor()

        print(f"Index optimization started: {index_name} ...")

        reindex_start = time.time()
        cur.execute(f"REINDEX INDEX {index_name};")
        conn.commit()
        reindex_end = time.time()
        reindex_time = reindex_end - reindex_start
        print(f"REINDEX completed! (Elapsed time: {reindex_time:.2f} seconds)")

        vacuum_start = time.time()
        cur.execute(f"VACUUM ANALYZE {table_name};")
        conn.commit()
        vacuum_end = time.time()
        vacuum_time = vacuum_end - vacuum_start
        print(f"✅ VACUUM ANALYZE completed! (Elapsed time: {vacuum_time:.2f} seconds)")

        cur.close()
        conn.close()

    def get_pgvector_info(self):
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        table_name = f"image_embeddings_{self.image_embedding_model_name}"

        cur.execute(f"SELECT DISTINCT(code) FROM {table_name}")
        rows = cur.fetchall()
        indexed_img_codes = [row[0] for row in rows]

        cur.execute(f"SELECT COUNT(DISTINCT id) FROM {table_name};")
        unique_id_count = cur.fetchone()[0]

        cur.execute(f"SELECT MIN(id), MAX(id) FROM {table_name};")
        min_id, max_id = cur.fetchone()

        cur.close()
        conn.close()

        return {
            "num_of_unique_image_embeddings": unique_id_count,
            "min_id": min_id,
            "max_id": max_id,
            "indexed_codes": indexed_img_codes
        }

    def insert_image_embeddings_into_postgres(self, batch_ids, batch_img_codes, image_embeddings, batch_cats):
        self.create_index()
        self.insert_embeddings(batch_ids, batch_img_codes, image_embeddings, batch_cats)

    def search_similar_vectors(self, query_ids, query_embeddings, query_categories):
        table_name = f"image_embeddings_{self.image_embedding_model_name}"
        config = self.config

        conn = connect_db(config)
        cur = conn.cursor()

        cur.execute(f"SET hnsw.ef_search = 200;")
        cur.execute(f"SET enable_seqscan = off;")
        select_clause = f"""
            SELECT id, category, embedding <=> %s::vector AS distance
            FROM {table_name}
        """

        all_ids = []
        all_cats = []
        all_scores = []

        for idx, (query_id, query_embedding, query_cat) in enumerate(
                zip(query_ids, query_embeddings, query_categories)):
            '''
            if query_cat == "":
                query = select_clause + "ORDER BY distance ASC LIMIT 10;"
                params = (query_embedding.tolist(),)
                # label = f"🔎 [전체 검색] - Query #{idx + 1}: {query_id}"
            else:
                query = select_clause + "WHERE category = %s ORDER BY distance ASC LIMIT 10;"
                params = (query_embedding.tolist(), query_cat)
                # label = f"🔍 [카테고리 필터 검색] - Query #{idx + 1}: {query_id}"
            '''
            query = select_clause + "ORDER BY distance ASC LIMIT 10;"
            params = (query_embedding.tolist(),)
            # label = f"🔎 [전체 검색] - Query #{idx + 1}: {query_id}"

            start_time = time.perf_counter()
            cur.execute(query, params)
            results = cur.fetchall()
            end_time = time.perf_counter()
            '''
            print(f"\n{label} Top 10 (by distance)")
            for i, (id_, cat, dist) in enumerate(results):
                print(f"{i + 1}. ID: {id_}, Cat: {cat}, Distance: {dist:.6f}")
            print(f"⏱️ 검색 소요 시간: {end_time - start_time:.4f}초")
            '''
            """
            cur.execute("EXPLAIN ANALYZE " + query, params)
            plan = cur.fetchall()
            print("\n📊 Execution plan:")
            for line in plan:
                print(line[0])
            """
            ids = [row[0] for row in results]
            all_ids.append(ids)
            cats = [row[1] for row in results]
            all_cats.append(cats)
            scores = [1 - row[2] for row in results]
            all_scores.append(scores)

        cur.close()
        conn.close()

        return all_ids, all_cats, all_scores

    def create_search_results_table(self):
        table_name = "search_results"
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                ymdh INTEGER,
                model_name VARCHAR(30),
                query_id VARCHAR(128),
                pu_id INTEGER,
                place_id INTEGER,
                c_key CHAR(32),
                au_id INTEGER,
                p_key VARCHAR(128),
                p_category VARCHAR(64),
                similarity FLOAT,
                category VARCHAR(128),
                bbox VARCHAR(128),
                bbox_size FLOAT,
                bbox_centrality FLOAT,
                p_score FLOAT
            );
        """)
        conn.commit()
        cur.close()
        conn.close()

    def insert_search_results(self, segment_id, result_ids, similarities, category, bbox, bbox_size, bbox_centrality):
        table_name = "search_results"
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()

        def get_p_category(segment_id):
            try:
                num = str(segment_id).split('_')[1]
                if num.isdigit():
                    return f"Object-{num}"
                else:
                    return f"Object-0"
            except Exception:
                return "Object-0"

        def get_base_query_id(segment_id):
            return str(segment_id).split('_', 1)[0]

        def get_p_key(result_id):
            parts = str(result_id).split('_')
            if len(parts) > 1:
                return '_'.join(parts[:-1])
            else:
                return str(result_id)

        p_category = get_p_category(segment_id)
        yolo_version = self.get_yolo_version()

        try:
            for result_id, similarity in zip(result_ids, similarities):
                now = datetime.now()
                ymdh = int(now.strftime('%Y%m%d%H'))
                p_key = get_p_key(result_id)
                query_id_db = get_base_query_id(segment_id)
                model_name = f"{self.image_embedding_model_name}_yolo{yolo_version}"
                query = f"""
                    INSERT INTO {table_name} (ymdh, model_name, query_id, category, p_key, p_category, similarity, bbox, bbox_size, bbox_centrality)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cur.execute(query,
                            (ymdh, model_name, query_id_db, category, p_key, p_category,
                             float(similarity), str(bbox), bbox_size, bbox_centrality))
            conn.commit()
        except Exception as e:
            print(f"Error inserting search results: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def update_search_results_columns(self, id, pu_id, place_id, c_key, au_id):
        table_name = "search_results"
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        try:
            query = f"""
                UPDATE {table_name}
                SET pu_id = %s, place_id = %s, c_key = %s, au_id = %s
                WHERE id = %s
            """
            cur.execute(query, (pu_id, place_id, c_key, au_id, id))
            conn.commit()
        except Exception as e:
            print(f"Error updating search results columns: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def update_p_score_column(self):
        table_name = "search_results"
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        try:
            # p_score = bbox_size * bbox_centrality * similarity
            query = f"""
                UPDATE {table_name}
                SET p_score = bbox_size * bbox_centrality * similarity
                WHERE bbox_size IS NOT NULL AND bbox_centrality IS NOT NULL AND similarity IS NOT NULL
            """
            cur.execute(query)
            conn.commit()
            print("[INFO] p_score column updated successfully.")
        except Exception as e:
            print(f"Error updating p_score column: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def get_rows_with_missing_columns(self):
        table_name = "search_results"
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        try:
            query = f"""
                SELECT id, query_id, p_key
                FROM {table_name}
                WHERE pu_id IS NULL OR place_id IS NULL OR c_key IS NULL OR au_id IS NULL
            """
            cur.execute(query)
            rows = cur.fetchall()  # [(id, query_id, p_key), ...]
            return rows
        except Exception as e:
            print(f"Error fetching rows with missing columns: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def get_search_results_id_range(self):
        table_name = "search_results"
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        try:
            query = f"SELECT MIN(id), MAX(id) FROM {table_name}"
            cur.execute(query)
            result = cur.fetchone()
            if result and result[0] is not None and result[1] is not None:
                return result[0], result[1]
            else:
                return None
        except Exception as e:
            print(f"Error fetching id range: {e}")
            return None
        finally:
            cur.close()
            conn.close()

    def save_top30_per_query_id(self, model_name=None):
        table_name = "search_results"
        top_table_name = "search_results_top30"
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        try:
            # 1. Create search_results_top30 table (id SERIAL PRIMARY KEY, other columns same)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {top_table_name} (
                    id SERIAL PRIMARY KEY,
                    ymdh INTEGER,
                    model_name VARCHAR(30),
                    query_id VARCHAR(128),
                    pu_id INTEGER,
                    place_id INTEGER,
                    c_key CHAR(32),
                    au_id INTEGER,
                    p_key VARCHAR(128),
                    p_category VARCHAR(64),
                    p_score FLOAT,
                    category VARCHAR(128),
                    bbox VARCHAR(128)
                );
            """)
            conn.commit()
            yolo_version = self.get_yolo_version()
            model_name_with_yolo_version = f"{model_name}_yolo{yolo_version}"
            # Extract list of (model_name, query_id)
            if model_name:
                cur.execute(f"SELECT DISTINCT model_name, query_id FROM {table_name} WHERE model_name = %s",
                            (model_name_with_yolo_version,))
            else:
                cur.execute(f"SELECT DISTINCT model_name, query_id FROM {table_name}")
            model_query_ids = cur.fetchall()  # [(model_name, query_id), ...]
            for model_name_val, qid in tqdm(model_query_ids, desc="Saving top30 per (model_name, query_id)"):
                # For each (model_name, query_id, p_key), keep only the row with the highest p_score, then select top 30
                cur.execute(f"""
                    INSERT INTO {top_table_name}
                    (ymdh, model_name, query_id, pu_id, place_id, c_key, au_id, p_key, p_category, p_score, category, bbox)
                    SELECT ymdh, model_name, query_id, pu_id, place_id, c_key, au_id, p_key, p_category, p_score, category, bbox
                    FROM (
                        SELECT DISTINCT ON (model_name, query_id, p_key) *
                        FROM {table_name}
                        WHERE model_name = %s AND query_id = %s
                        ORDER BY model_name, query_id, p_key, p_score DESC
                    ) AS sub
                    ORDER BY p_score DESC
                    LIMIT 30
                """, (model_name_val, qid))
            conn.commit()
            print(
                f"[INFO] Top 30 results per (model_name, query_id) saved to search_results_top30 table (duplicates removed)")
        except Exception as e:
            print(f"Error saving top30 per query_id: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()
    
    def get_search_results(self, model_name=None):
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        try:
            if model_name:
                cur.execute("""
                    SELECT model_name, pu_id, place_id, c_key, au_id, p_key, p_category, p_score, category, bbox
                    FROM search_results
                    WHERE model_name = %s
                    ORDER BY id
                """, (model_name,))
            else:
                cur.execute("""
                    SELECT model_name, pu_id, place_id, c_key, au_id, p_key, p_category, p_score, category, bbox
                    FROM search_results
                    ORDER BY id
                """)
            rows = cur.fetchall()
            return rows
        except Exception as e:
            print(f"Error fetching id range: {e}")
            return None
        finally:
            cur.close()
            conn.close()

    def get_search_results_30(self, model_name=None):
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        yolo_version = self.get_yolo_version()
        model_name_with_yolo_version = f"{model_name}_yolo{yolo_version}"
        try:
            if model_name:
                cur.execute("""
                    SELECT model_name, pu_id, place_id, c_key, au_id, p_key, p_category, p_score, category, bbox
                    FROM search_results_top30
                    WHERE model_name = %s
                    ORDER BY id
                """, (model_name_with_yolo_version,))
            else:
                cur.execute("""
                    SELECT model_name, pu_id, place_id, c_key, au_id, p_key, p_category, p_score, category, bbox
                    FROM search_results_top30
                    ORDER BY id
                """)
            rows = cur.fetchall()
            return rows
        except Exception as e:
            print(f"Error fetching search_results_top30: {e}")
            return None
        finally:
            cur.close()
            conn.close()

    def remove_duplicates_from_search_results(self):
        """
        search_results 테이블에서 (c_key, p_key, pu_id, au_id, model_name, p_category) 기준으로 similarity가 가장 높은 row만 남기고 나머지는 삭제합니다.
        """
        table_name = "search_results"
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        try:
            query = f"""
            WITH ranked AS (
              SELECT
                id,
                ROW_NUMBER() OVER (
                  PARTITION BY c_key, p_key, pu_id, au_id, model_name, p_category
                  ORDER BY similarity DESC
                ) AS rn
              FROM {table_name}
            )
            DELETE FROM {table_name}
            USING ranked
            WHERE {table_name}.id = ranked.id
              AND ranked.rn > 1;
            """
            cur.execute(query)
            conn.commit()
            print("[INFO] Duplicates removed from search_results.")
        except Exception as e:
            print(f"Error removing duplicates: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()
