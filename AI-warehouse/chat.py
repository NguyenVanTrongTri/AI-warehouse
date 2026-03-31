import joblib
import json
import re
import unicodedata
import pandas as pd
import duckdb
import subprocess
import sys
import os
import io
import gc
from datetime import datetime
from predict import LoraPredictor

# --- CẤU HÌNH HỆ THỐNG ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

predictor = LoraPredictor()

# --- BIẾN TOÀN CỤC ---
MODEL_DIR = "model"
DYNAMIC_RESPONSES = {}
model = None
vectorizer = None
resources_loaded = False 

def load_resources():
    global DYNAMIC_RESPONSES, model, vectorizer, resources_loaded
    if resources_loaded:
        return
    try:
        print("[Hệ thống] Đang nạp Model và Snapshot...")
        model = joblib.load(os.path.join(MODEL_DIR, "warehouse_model.pkl"))
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        with open(os.path.join(MODEL_DIR, "intent_map.json"), "r", encoding="utf-8") as f:
            DYNAMIC_RESPONSES = json.load(f)
        
        snapshot_path = "data_history/database_snapshot.bin"
        if os.path.exists(snapshot_path):
            db_snapshot = joblib.load(snapshot_path)
            for table_name, data in db_snapshot["tables"].items():
                globals()[table_name] = pd.DataFrame(data)
            del db_snapshot
            gc.collect() 
            
        resources_loaded = True
        print("[Hệ thống] Nạp tài nguyên hoàn tất.")
    except Exception as e:
        print(f"[Lỗi Resource] {e}")

# --- HELPER FUNCTIONS ---
def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    result = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return result.replace('đ', 'd').replace('Đ', 'D')

def clean_text(text):
    if not text: return ""
    text = text.lower()
    text = remove_accents(text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())

def execute_query_on_bin(sql_template, user_input):
    if not resources_loaded:
        load_resources()
        
    final_sql = sql_template
    column_translator = {
        "MaHangHoa": "Mã hàng", "DonViTinh": "Đơn vị tính",
        "TenHangHoa": "Tên mặt hàng", "SoLuong": "Số lượng",
        "GiaBan": "Giá bán", "MaPhieuNhap": "Số phiếu",
        "is_new": "Trạng thái"
    }

    if ":param" in sql_template:
        match = re.search(r'([A-Z]*\d+)', user_input.upper())
        param = match.group(1) if match else "UNKNOWN"
        final_sql = sql_template.replace(":param", param)

    # Logic lọc is_new = 1
    sql_low = final_sql.lower()
    if "where" in sql_low:
        if "order by" in sql_low: final_sql = re.sub(r"(?i)order by", "AND is_new = 1 ORDER BY", final_sql)
        else: final_sql += " AND is_new = 1"
    else:
        if "group by" in sql_low: final_sql = re.sub(r"(?i)group by", "WHERE is_new = 1 GROUP BY", final_sql)
        else: final_sql += " WHERE is_new = 1"

    try:
        result_df = duckdb.query(final_sql).to_df()
        if result_df.empty:
            return "Hiện tại không có dữ liệu mới nào từ Web khớp với yêu cầu."
        
        columns = [col for col in result_df.columns if col != 'is_new']
        rows = result_df[columns].values
        response_text = f"Kết quả mới cập nhật ({len(rows)} bản ghi):\n"
        for row in rows:
            row_parts = [f"{column_translator.get(columns[i], columns[i])}: {row[i]}" for i in range(len(columns))]
            response_text += " | ".join(row_parts) + "\n"
        return response_text
    except Exception as e:
        return f"LORA AI: Lỗi truy vấn dữ liệu ({e})"

def get_answer(intent, user_input, confidence, dynamic_responses):
    if not resources_loaded:
        load_resources()
        
    if confidence < 0.20:
        return "Mình chưa hiểu ý bạn, Leader vui lòng nhập lại rõ hơn nhé."

    if intent in ["predict_import", "prediction_import_count"]:
        df_hh = globals().get('hanghoa')
        return predictor.handle_forecast_logic(df_hh)

    resps = dynamic_responses if dynamic_responses else DYNAMIC_RESPONSES
    response_data = resps.get(intent, "")
    
    if intent.startswith("sql_") or "SELECT" in response_data.upper():
        return execute_query_on_bin(response_data, user_input)
    
    return response_data

def log_to_results(user_input, intent, confidence, ai_response, dynamic_responses):
    # Hàm log giữ nguyên để tránh lỗi undefined
    folder_path = "dataset"
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    # Logic log của ông ở đây...
    pass

# --- CHẾ ĐỘ CHẠY TERMINAL ---
if __name__ == "__main__":
    load_resources()
    print("\n[HỆ THỐNG] LORA AI LEVEL 3 ĐÃ SẴN SÀNG (TERMINAL MODE)!") 

    while True:
        try:
            query = input("Bạn: ").strip()
            if not query: continue
            if query.lower() in ['exit', 'quit', 'thoát']: break
            
            cleaned = clean_text(query)
            X = vectorizer.transform([cleaned])
            intent = model.predict(X)[0]
            confidence = max(model.predict_proba(X)[0])
            
            answer = get_answer(intent, query, confidence, DYNAMIC_RESPONSES)
            print(f"LORA AI: {answer}\n")
            log_to_results(query, intent, confidence, answer, DYNAMIC_RESPONSES)
        except KeyboardInterrupt:
            break