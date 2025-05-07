import psycopg2
import time


def connect_db(config):
    try:
        return psycopg2.connect(**config["database"])
    except Exception as e:
        raise ValueError(f"Error connecting to database: {e}")


class PGVectorDB:
    def __init__(self, image_embedding_model_name, config):
        conn = connect_db(config)

        model_config = config["model"][image_embedding_model_name]
        table_name = f"image_embeddings_{image_embedding_model_name}"
        num_dimensions = model_config["num_dimension"]

        cur = conn.cursor()
        try:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id VARCHAR(255) PRIMARY KEY,
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

        self.image_embedding_model_name = image_embedding_model_name
        self.config = config

    def insert_embeddings(self, ids, image_embeddings, cats):
        table_name = f"image_embeddings_{self.image_embedding_model_name}"

        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        try:
            for id, embedding, cat in zip(ids, image_embeddings, cats):
                query = f"""
                    INSERT INTO {table_name} (id, embedding, category)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                """
                embedding = embedding.tolist()
                cur.execute(query, (str(id), embedding, cat))
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

        print(f"색인 최적화 시작: {index_name} ...")

        reindex_start = time.time()
        cur.execute(f"REINDEX INDEX {index_name};")
        conn.commit()
        reindex_end = time.time()
        reindex_time = reindex_end - reindex_start
        print(f"REINDEX 완료! (소요 시간: {reindex_time:.2f}초)")

        vacuum_start = time.time()
        cur.execute(f"VACUUM ANALYZE {table_name};")
        conn.commit()
        vacuum_end = time.time()
        vacuum_time = vacuum_end - vacuum_start
        print(f"✅ VACUUM ANALYZE 완료! (소요 시간: {vacuum_time:.2f}초)")

        cur.close()
        conn.close()

    def get_pgvector_info(self):
        config = self.config
        conn = connect_db(config)
        cur = conn.cursor()
        table_name = f"image_embeddings_{self.image_embedding_model_name}"

        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(DISTINCT id) FROM {table_name};")
        unique_id_count = cur.fetchone()[0]

        cur.execute(f"SELECT MIN(id), MAX(id) FROM {table_name};")
        min_id, max_id = cur.fetchone()

        cur.close()
        conn.close()

        return {
            "num_of_total_image_embeddings": row_count,
            "num_of_unique_image_embeddings": unique_id_count,
            "min_id": min_id,
            "max_id": max_id
        }

    def insert_image_embeddings_into_postgres(self, batch_ids, image_embeddings, batch_cats):
        self.create_index()
        self.insert_embeddings(batch_ids, image_embeddings, batch_cats)

    def search_similar_vectors(self, query_ids, query_embeddings, category):
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
        all_distances = []

        for idx, (query_id, query_embedding) in enumerate(zip(query_ids, query_embeddings)):
            if category is None:
                query = select_clause + "ORDER BY distance ASC LIMIT 10;"
                params = (query_embedding.tolist(),)
                label = f"🔎 [전체 검색] - Query #{idx + 1}: {query_id}"
            else:
                query = select_clause + "WHERE category = %s ORDER BY distance ASC LIMIT 10;"
                params = (query_embedding.tolist(), category)
                label = f"🔍 [카테고리 필터 검색] - Query #{idx + 1}: {query_id}"

            start_time = time.perf_counter()
            cur.execute(query, params)
            results = cur.fetchall()
            end_time = time.perf_counter()

            print(f"\n{label} Top 10 (by distance)")
            for i, (id_, cat, dist) in enumerate(results):
                print(f"{i + 1}. ID: {id_}, Cat: {cat}, Distance: {dist:.6f}")
            print(f"⏱️ 검색 소요 시간: {end_time - start_time:.4f}초")

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
            distances = [row[2] for row in results]
            all_distances.append(distances)

        cur.close()
        conn.close()

        return all_ids, all_cats, all_distances
