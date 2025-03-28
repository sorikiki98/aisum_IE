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
root_dir = "./data"
os.makedirs(root_dir, exist_ok=True)
try:
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute(
        "SELECT img_url, category1_code, category2_code from piclick.product_list2 where au_id=6318 and site_id=2775")
    img_tuples = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
    for i, t in enumerate(img_tuples):
        try:
            url, cat1, cat2 = t
            response = requests.get(url, stream=True)
            response.raise_for_status()
            file_name = os.path.splitext(os.path.basename(url))[0]
            file_extension = os.path.splitext(os.path.basename(url))[1]
            save_path = os.path.join(f"{root_dir}/eseltree/images/{cat1}/{cat2}", f"{file_name}{file_extension}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
