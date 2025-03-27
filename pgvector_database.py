import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from torch import nn
from PIL import Image
import os
import random
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm
import time
from image_embedding_model import get_num_dimensions_of_image_embedding_model

DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "0000",
    "host": "localhost",
    "port": "5432",
}


def connect_db():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise


def create_table(image_embedding_model_name):
    table_name = f"image_embeddings_{image_embedding_model_name}"
    num_dimensions = get_num_dimensions_of_image_embedding_model(image_embedding_model_name)

    conn = connect_db()
    cur = conn.cursor()

    try:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id VARCHAR(255) PRIMARY KEY,
                embedding VECTOR({num_dimensions}),
                category1 VARCHAR(10),
                category2 VARCHAR(10)
            );
        """)
        conn.commit()
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def insert_embeddings(image_embedding_model_name, ids, image_embeddings, cat1s, cat2s):
    table_name = f"image_embeddings_{image_embedding_model_name}"

    conn = connect_db()
    cur = conn.cursor()
    try:
        for id, embedding, cat1, cat2 in zip(ids, image_embeddings, cat1s, cat2s):
            print(id, embedding[:3])
            query = f"""
                INSERT INTO "{table_name}" (id, embedding, category1, category2)
                VALUES (%s, %s, %s, %s)
            """
            embedding = embedding.tolist()
            cur.execute(query, (str(id), embedding, cat1, cat2))
        conn.commit()

    except Exception as e:
        print(f"Error inserting into PostgreSQL: {e}")
        conn.rollback()

    finally:
        cur.close()
        conn.close()


def create_index(image_embedding_model_name):
    table_name = f"image_embeddings_{image_embedding_model_name}"

    conn = connect_db()
    cur = conn.cursor()

    try:
        cur = conn.cursor()
        index_query = f"""
            CREATE INDEX IF NOT EXISTS hnsw_idx_{image_embedding_model_name}
            ON "{table_name}" USING hnsw (embedding vector_cosine_ops)
            WITH (m = 32, ef_construction = 300);
        """
        cur.execute(index_query)
        conn.commit()

        print(f"HNSW index created on {table_name}.")

    except Exception as e:
        print("Error creating HNSW index:")
        print(index_query)
        print("Error:", e)
        conn.rollback()

    finally:
        cur.close()
        conn.close()


def optimize_index(image_embedding_model_name):
    table_name = f"image_embeddings_{image_embedding_model_name}"
    index_name = f"hnsw_idx_{image_embedding_model_name}"

    conn = connect_db()
    cur = conn.cursor()

    print(f"📌 색인 최적화 시작: {index_name} ...")

    reindex_start = time.time()
    cur.execute(f"REINDEX INDEX {index_name};")
    conn.commit()
    reindex_end = time.time()
    reindex_time = reindex_end - reindex_start
    print(f"✅ REINDEX 완료! (소요 시간: {reindex_time:.2f}초)")

    vacuum_start = time.time()
    cur.execute(f"VACUUM ANALYZE {table_name};")
    conn.commit()
    vacuum_end = time.time()
    vacuum_time = vacuum_end - vacuum_start
    print(f"✅ VACUUM ANALYZE 완료! (소요 시간: {vacuum_time:.2f}초)")

    cur.close()
    conn.close()


def print_pgvector_info(image_embedding_model_name):
    conn = connect_db()
    cur = conn.cursor()
    table_name = f"image_embeddings_{image_embedding_model_name}"

    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(DISTINCT id) FROM {table_name};")
    unique_id_count = cur.fetchone()[0]

    cur.execute(f"SELECT MIN(id), MAX(id) FROM {table_name};")
    min_id, max_id = cur.fetchone()

    print("--- pgvector table information ---")
    print(f"Table: {table_name}")
    print(f"Row count (total records): {row_count}")
    print(f"Unique ID count: {unique_id_count}")
    print(f"Min ID: {min_id}")
    print(f"Max ID: {max_id}")
    print("-------------------------------")

    cur.close()
    conn.close()


def insert_image_embeddings_into_postgres(image_embedding_model_name, batch_ids, image_embeddings, batch_cat1s,
                                          batch_cat2s):
    create_table(image_embedding_model_name)
    create_index(image_embedding_model_name)
    insert_embeddings(image_embedding_model_name, batch_ids, image_embeddings, batch_cat1s, batch_cat2s)
    print_pgvector_info(image_embedding_model_name)


def search_similar_vectors(image_embedding_model_name, query_ids, query_embeddings, category1, category2):
    table_name = f"image_embeddings_{image_embedding_model_name}"

    conn = connect_db()
    cur = conn.cursor()

    cur.execute(f"SET hnsw.ef_search = 200;")
    cur.execute(f"SET enable_seqscan = off;")
    select_clause = f"""
        SELECT id, category1, category2, embedding <=> %s::vector AS distance
        FROM {table_name}
    """

    all_ids = []
    cat1s = []
    cat2s = []
    all_distances = []

    for idx, (query_id, query_embedding) in enumerate(zip(query_ids, query_embeddings)):
        if category1 is None or category2 is None:
            query = select_clause + "ORDER BY distance ASC LIMIT 10;"
            params = (query_embedding.tolist(),)
            label = f"🔎 [전체 검색] - Query #{idx + 1}: {query_id}"
        else:
            query = select_clause + "WHERE category1 = %s AND category2 = %s ORDER BY distance ASC LIMIT 10;"
            params = (query_embedding.tolist(), category1, category2)
            label = f"🔍 [카테고리 필터 검색] - Query #{idx + 1}: {query_id}"

        start_time = time.perf_counter()
        cur.execute(query, params)
        results = cur.fetchall()
        end_time = time.perf_counter()

        print(f"\n{label} Top 10 (by distance)")
        for i, (id_, cat1, cat2, dist) in enumerate(results):
            print(f"{i + 1}. ID: {id_}, Cat1: {cat1}, Cat2: {cat2}, Distance: {dist:.6f}")
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
        category1s = [row[1] for row in results]
        category2s = [row[2] for row in results]
        distances = [row[3] for row in results]
        all_distances.append(distances)

    cur.close()
    conn.close()

    return all_ids, category1s, category2s, all_distances
