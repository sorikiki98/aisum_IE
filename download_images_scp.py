import os
import subprocess
import pymysql

# loca_base 경로 확인
server_ip = "115.68.199.75"
server_user = "root"
password = "psr1127!"
remote_base = "/home/piclick/piclick.tmp/ITEM_IMG/AIPIC_KR"
local_base = "/mnt/e/AIPIC_KR"

mysql_config = {
    "host": "db.main.piclick.kr",
    "port": 3306,
    "user": "piclick",
    "password": "psr9566!",
    "database": "piclick",
    "read_timeout": 300,
    "write_timeout": 300,
    "connect_timeout": 300,
    "charset": "utf8mb4"
}

# 드라이브 마운트
def ensure_drive_mounted(drive_letter="e"):
    mount_point = f"/mnt/{drive_letter.lower()}"

    if os.path.ismount(mount_point):
        return

    print(f"드라이브 {drive_letter.upper()}: 마운트 시도 중...")

    try:
        subprocess.run(["sudo", "mkdir", "-p", mount_point], check=True)
        subprocess.run([
            "sudo", "mount", "-t", "drvfs", f"{drive_letter.upper()}:", mount_point
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ /mnt/{drive_letter.lower()} 마운트 실패: {e}")
        raise


# 서버에서 site_id 디렉터리 리스트 가져오기
def get_site_ids_from_server():
    try:
        result = subprocess.check_output(
            ["ssh", f"{server_user}@{server_ip}", f"ls {remote_base}"],
            stderr=subprocess.STDOUT,
            text=True
        )
        return sorted([s for s in result.strip().split("\n") if s.isdigit()])
    except subprocess.CalledProcessError as e:
        print("❌ SSH 접속 실패:", e.output)
        return []

# MySQL에서 status=1인 site_id 목록만 추출
def get_status1_site_ids():
    connection = pymysql.connect(**mysql_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT site_id
                FROM piclick.product_list2
                WHERE status = 1
            """)
            return set(str(row[0]) for row in cursor.fetchall())
    finally:
        connection.close()


# MySQL에서 남겨야 할 이미지 경로 가져오기 (status=1)
def get_keep_paths(site_id):
    connection = pymysql.connect(**mysql_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT save_path, save_name FROM piclick.product_list2
                WHERE site_id = {site_id} AND status = 1
            """)
            keep_set = set()
            for save_path, save_name in cursor.fetchall():
                # save_path: 'AIPIC_KR/103/230511'
                # rel_path: '103/230511/filename.jpg'
                rel_path = os.path.join(*save_path.split("/")[1:], save_name)
                keep_set.add(rel_path)
            return keep_set
    finally:
        connection.close()

# 다운로드
def scp_download(site_id):
    local_site_path = os.path.join(local_base, site_id)
    if os.path.exists(local_site_path):
        print(f"이미 존재함: {site_id} → 다운로드 생략")
        return
    
    os.makedirs(os.path.dirname(local_site_path), exist_ok=True)
    
    remote_path = f"{server_user}@{server_ip}:{remote_base}/{site_id}"
    print(f"다운로드 중: site_id={site_id}")
    subprocess.run([
        "sshpass", "-p", password,
        "scp", "-r", remote_path, local_site_path
    ], check=True)

# save_path 기준으로 이미지 정리
def clean_by_path(site_id, keep_paths):
    local_site_path = os.path.join(local_base, site_id)
    deleted_count = 0

    for root, _, files in os.walk(local_site_path):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, start=local_base)
            if rel_path not in keep_paths:
                try:
                    os.remove(abs_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 삭제 실패: {rel_path} - {e}")

    remaining_files = sum(len(files) for _, _, files in os.walk(local_site_path))
    return deleted_count, remaining_files

def get_existing_paths(site_id):
    local_site_path = os.path.join(local_base, site_id)
    existing = set()
    for root, _, files in os.walk(local_site_path):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, start=local_base)
            existing.add(rel_path)
    return existing

def compare_server_and_db_sites(server_site_ids, db_site_ids):
    only_in_server = sorted(server_site_ids - db_site_ids)
    only_in_db = sorted(db_site_ids - server_site_ids)

    print(f"\nDB에는 있지만 서버에 없는 site_id 수: {len(only_in_db)}개")
    print(f"목록: {', '.join(only_in_db[:10])}...")
    
    with open("only_in_server.txt", "w") as f:
        f.write("\n".join(only_in_server))
    with open("only_in_db.txt", "w") as f:
        f.write("\n".join(only_in_db))

if __name__ == "__main__":
    #드라이브 인식안되면 아래 코드 활성화
    # ensure_drive_mounted("e")

    with open("failed_log.txt", "w", encoding="utf-8") as f:
        f.write("site_id,DB_expected,actual_remaining,total_downloaded,missing_file_count,missing_files\n")

    server_site_ids = set(get_site_ids_from_server())
    status1_site_ids = get_status1_site_ids()
    site_ids = sorted(server_site_ids & status1_site_ids)
    print(f"처리 대상 site_id 수: {len(site_ids)}개")
    compare_server_and_db_sites(server_site_ids, status1_site_ids)
    print("download start!")

    for site_id in site_ids:
        try:
            keep_paths = get_keep_paths(site_id)
            scp_download(site_id)
            deleted_count, remaining_files = clean_by_path(site_id, keep_paths)
            total_downloaded = deleted_count + remaining_files

            existing_paths = get_existing_paths(site_id)
            missing_paths = keep_paths - existing_paths
            missing_count = len(missing_paths)
            missing_files_str = ";".join(str(p) for p in sorted(missing_paths)) if missing_paths else ""
            print("--------------------------------")
            print(f"총 다운로드된 파일 수: {total_downloaded}개")
            print(f"삭제된 이미지 수: {deleted_count}개")
            print(f"실제 남은 이미지 수: {remaining_files}개")
            print(f"DB 기준 유지할 파일 수: {len(keep_paths)}개")
            print("--------------------------------")
            if missing_paths:
                print(f"DB 대상 파일 중 서버에 존재하지 않는 파일: {missing_count}개")

            if remaining_files != len(keep_paths) or missing_count > 0:
                print(f"검증 실패: {site_id} (DB: {len(keep_paths)}, 실제: {remaining_files})")
                with open("failed_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"{site_id},{len(keep_paths)},{remaining_files},{total_downloaded},{missing_count},{missing_files_str}\n")
            else:
                print(f"검증 통과: {site_id}")
        
        except Exception as e:
            print(f"❌ site_id {site_id} 처리 중 에러: {e}")
            with open("failed_log.txt", "a", encoding="utf-8") as f:
                f.write(f"{site_id},ERROR,{str(e).replace(',', ' ')},0,0,\n")