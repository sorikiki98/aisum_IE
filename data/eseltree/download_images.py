import pymysql
import os
import requests

db_config = {
    "host": "db.main.piclick.kr",
    "port": 3306,
    "user": "piclick",
    "password": "psr9566!",
    "db": "piclick",
    "read_timeout": 300,
    "write_timeout": 300,
    "connect_timeout": 300,
}
save_dir = "images"
os.makedirs(save_dir, exist_ok=True)
try:
    # MySQL 연결
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()
    # img_url 가져오기
    cursor.execute("SELECT img_url FROM piclick.product_list2 WHERE au_id = 6318 AND site_id = 2775")
    img_urls = [row[0] for row in cursor.fetchall()]
    for i, url in enumerate(img_urls):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            file_name = os.path.splitext(os.path.basename(url))[0]
            file_extension = os.path.splitext(os.path.basename(url))[1]
            save_path = os.path.join(save_dir, f"{file_name}{file_extension}")
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Download Completed: {save_path}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Download Failed ({url}): {e}")
    cursor.close()
    connection.close()
except pymysql.MySQLError as e:
    print(f"❌ DB Connection Failed: {e}")
