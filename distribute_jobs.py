#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymysql
import subprocess
import math
from datetime import datetime
import os
import shutil
import sys
import json 


# config.json 파일 로드
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: config.json 파일을 찾을 수 없습니다.")
    sys.exit(1)

mysql_config = config["database"]["mysql"]

if 'db' in mysql_config:
    mysql_config['database'] = mysql_config.pop('db')

PATHS = config.get("paths", {})
LOG_FILE = PATHS.get("log_file", "/home/ec2-user/aisum_IE/log/distribute.log")
SLAVE_TASK_PATH = PATHS.get("slave_task_path", "/home/ec2-user/aisum_IE/task/")
SCP_KEY_PATH = PATHS.get("scp_key_path", "/home/ec2-user/pm-server-key.pem")

SLAVE_USER = "ec2-user" 
SLAVE_IP = "172.16.10.209"

def log(message):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

def distribute_embedding_jobs():
    log("===== 임베딩 작업 분배 시작 =====")
    db_conn = None
    try:
        local_task_dir = "/home/ec2-user/aisum_IE/task" 

        db_conn = pymysql.connect(**mysql_config) 
        with db_conn.cursor() as cursor:
            db_name = config["database"]["mysql"].get("db", "test")
            table_name = config["database"]["mysql"].get("table", "product_list")

            sql_select = f"SELECT p_key FROM {db_name}.{table_name} WHERE status = 2 AND img_dn = 'D' LIMIT 100000;"
            cursor.execute(sql_select)
            p_keys_to_process = [row[0] for row in cursor.fetchall()]

            if not p_keys_to_process:
                log("새로운 임베딩 대상이 없습니다.")
                return

            p_keys_str = ','.join([f"'{key}'" for key in p_keys_to_process])
            sql_lock = f"UPDATE {db_name}.{table_name} SET img_dn = 'E' WHERE p_key IN ({p_keys_str});"
            cursor.execute(sql_lock)
            db_conn.commit()
            log(f"총 {len(p_keys_to_process)}개 작업을 'E' 상태로 잠갔습니다.")

            # 작업 분배
            total_jobs = len(p_keys_to_process)
            split_point = math.ceil(total_jobs / 2)
            master_p_keys = p_keys_to_process[:split_point]
            slave_p_keys = p_keys_to_process[split_point:]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if slave_p_keys:
                slave_filename = f"slave_tasks_{timestamp}.txt"
                temp_slave_filepath = os.path.join(script_dir, slave_filename)
                with open(temp_slave_filepath, "w") as f:
                    for p_key in slave_p_keys:
                        f.write(f"{p_key}\n")
                
                scp_command = [
                    "scp", "-i", SCP_KEY_PATH,
                    temp_slave_filepath,
                    f"{SLAVE_USER}@{SLAVE_IP}:{SLAVE_TASK_PATH}{slave_filename}"
                ]
                subprocess.run(scp_command, check=True, timeout=60)
                os.remove(temp_slave_filepath) 
                log(f"Slave용 작업 파일 '{slave_filename}'을 {SLAVE_IP} 서버로 전송 완료.")

            if master_p_keys:
                master_filename = f"master_tasks_{timestamp}.txt"
                temp_master_filepath = os.path.join(script_dir, master_filename)
                with open(temp_master_filepath, "w") as f:
                    for p_key in master_p_keys:
                        f.write(f"{p_key}\n")
                
                final_master_filepath = os.path.join(local_task_dir, master_filename)
                shutil.move(temp_master_filepath, final_master_filepath)
                log(f"Master용 작업 파일 '{master_filename}'을 'task' 폴더로 이동 완료.")

    except Exception as e:
        log(f"❌ 스크립트 실행 중 에러 발생: {e}")
    finally:
        if db_conn:
            db_conn.close()
        log("===== 임베딩 작업 분배 종료 =====")

if __name__ == "__main__":
    distribute_embedding_jobs()