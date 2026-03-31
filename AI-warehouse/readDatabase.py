import mysql.connector
import joblib
import os
import gc
from datetime import datetime

_CACHED_SNAPSHOT = None 

def read_data():
    global _CACHED_SNAPSHOT
    try:
        # Tự động xác định đường dẫn tuyệt đối để chạy được trên mọi môi trường
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(base_dir, 'data_history')
        target_file = os.path.join(target_dir, "database_snapshot.bin")
        
        # 1. LOAD BAN ĐẦU (SỬA LỖI KEY ERROR)
        if _CACHED_SNAPSHOT is None:
            if os.path.exists(target_file):
                loaded_data = joblib.load(target_file)
                # Kiểm tra nếu file cũ không có cấu trúc chuẩn thì reset lại
                if isinstance(loaded_data, dict) and 'tables' in loaded_data:
                    _CACHED_SNAPSHOT = loaded_data
                    total_rows = sum(len(rows) for rows in _CACHED_SNAPSHOT['tables'].values())
                    print(f">>> [HỆ THỐNG]: Đã nạp {total_rows} dòng dữ liệu lịch sử.")
                else:
                    print(">>> [CẢNH BÁO]: Snapshot cũ lỗi cấu trúc. Khởi tạo lại...")
                    _CACHED_SNAPSHOT = {"tables": {}, "schema": {}, "query_logic": {}}
            else:
                _CACHED_SNAPSHOT = {"tables": {}, "schema": {}, "query_logic": {}}

        # 2. KẾT NỐI DATABASE (Sử dụng config của ông)
        config = {
            'host': "db-bach-hoa-xanh-trongtri14780-2a54.j.aivencloud.com",
            'port': "16063",
            'user': "avnadmin",
            'password': "AVNS_0jfSoEV9FoIcW9MljSE",
            'database': "db_quanlykho",
            # Nếu chạy trên Render, hãy thêm dòng này và file ca.pem
            # 'ssl_ca': os.path.join(base_dir, "ca.pem"),
            # 'ssl_verify_cert': True
        }

        conn = mysql.connector.connect(**config)
        cursor = conn.cursor(dictionary=True, buffered=True)

        cursor.execute("SHOW TABLES")
        tables = [list(row.values())[0] for row in cursor.fetchall()]

        has_new_data = False
        updated_tables = []
        total_new_rows = 0

        for table in tables:
            # Tìm Primary Key
            cursor.execute(f"SHOW KEYS FROM {table} WHERE Key_name = 'PRIMARY'")
            pk_info = cursor.fetchone()
            id_col = pk_info['Column_name'] if pk_info else None
            
            if not id_col: continue

            # Đảm bảo table tồn tại trong dict
            if table not in _CACHED_SNAPSHOT["tables"]:
                _CACHED_SNAPSHOT["tables"][table] = []
            
            existing_rows = _CACHED_SNAPSHOT["tables"][table]
            is_first_time = len(existing_rows) == 0

            # Tìm mốc ID lớn nhất
            max_val = 0
            if not is_first_time:
                try:
                    max_val = max(int(row.get(id_col, 0)) for row in existing_rows)
                except:
                    max_val = existing_rows[-1].get(id_col, 0)

            # Lấy dữ liệu mới
            query = f"SELECT * FROM {table} WHERE {id_col} > %s" if not is_first_time else f"SELECT * FROM {table}"
            cursor.execute(query, (max_val,) if not is_first_time else ())
            new_data = cursor.fetchall()

            if new_data:
                assigned_flag = 0 if is_first_time else 1
                for row in new_data:
                    row['is_new'] = assigned_flag 
                    row['sync_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                _CACHED_SNAPSHOT["tables"][table].extend(new_data)
                print(f"    [+] {table.ljust(20)}: +{len(new_data)} dòng (is_new={assigned_flag}).")
                has_new_data = True
                total_new_rows += len(new_data)
                updated_tables.append(table)

        # 3. LƯU LẠI
        if has_new_data:
            if not os.path.exists(target_dir): os.makedirs(target_dir)
            joblib.dump(_CACHED_SNAPSHOT, target_file, compress=3) # Nén để tiết kiệm RAM
            print(f">>> [SUCCESS]: CẬP NHẬT HOÀN TẤT! (+{total_new_rows} dòng)")
        else:
            print(">>> [INFO]: Dữ liệu đã đồng bộ.")

        cursor.close()
        conn.close()
        gc.collect() # Giải phóng bộ nhớ
        return True

    except Exception as e:
        print(f"\n[LỖI]: {e}")
        return False

if __name__ == "__main__":
    read_data()