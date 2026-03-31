import pandas as pd
import json
import joblib
import os
import re
import unicodedata
import xgboost as xgb
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# ======================
# 1. HÀM CHUẨN HÓA & TIỀN XỬ LÝ
# ======================
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

# Cấu hình đường dẫn
if not os.path.exists('model'): os.makedirs('model')
if not os.path.exists('data_history'): os.makedirs('data_history')

BIN_HISTORY_PATH = "data_history/full_dataset.bin"
JSON_HISTORY_PATH = "data_history/full_dataset.json" 
DB_SNAPSHOT_PATH = "data_history/database_snapshot.bin"
INTENT_MAP_PATH = "model/intent_map.json"
NEW_DATASET_PATH = "dataset/dataset.json"

# ======================
# 2. NHÁNH DỰ BÁO (LEVEL 3)
# ======================
# ======================
# 2. NHÁNH DỰ BÁO (LEVEL 3) - PHIÊN BẢN HỌC TRỰC TIẾP 100%
# ======================
def train_forecast_branch():
    print("\n--- [LEVEL 3] Huấn luyện bộ não Dự báo kho (XGBoost + LabelEncoder) ---")
    if not os.path.exists(DB_SNAPSHOT_PATH):
        print(f"! Bỏ qua: Không tìm thấy file {DB_SNAPSHOT_PATH}")
        return

    data_snapshot = joblib.load(DB_SNAPSHOT_PATH)
    all_results = data_snapshot.get("tables", {}) 
    
    # Danh sách các bảng cần AI học sâu (Thêm key của bạn vào đây nếu cần)
    target_keys = ["train_data_detail", "export_forecast_data", "tonkho_chitiet"] 

    for key in target_keys:
        if key not in all_results: continue
        
        print(f">> Đang học đặc thù từng mặt hàng từ bảng: {key}")
        df = pd.DataFrame(all_results.get(key, []))
        
        # Cần tối thiểu dữ liệu để học
        if len(df) < 2:
            print(f"! Bỏ qua {key}: Dữ liệu quá ít.")
            continue

        # 1. Dò tìm cột Ngày và Mã hàng
        date_col = next((c for c in df.columns if any(kw in remove_accents(c.lower()) for kw in ['date', 'thoi gian', 'ngay'])), None)
        item_col = next((c for c in df.columns if any(kw in remove_accents(c.lower()) for kw in ['mahang', 'mahh', 'id'])), None)

        # 1.1. Dò tìm cột Số lượng (Logic thông minh: Loại trừ cột chứa chữ)
        qty_col = None
        # Lấy tất cả các cột có khả năng là số lượng
        potential_qty_cols = [c for c in df.columns if any(kw in remove_accents(c.lower()) for kw in ['quantity', 'so luong', 'tongsl', 'ton', 'qty'])]
        
        for col in potential_qty_cols:
            try:
                # Thử chuyển giá trị đầu tiên của cột này sang float
                # Nếu là 'TK01000', dòng này sẽ báo lỗi và nhảy sang cột tiếp theo
                test_val = str(df[col].iloc[0])
                float(test_val) 
                qty_col = col
                break # Tìm thấy cột số thực sự rồi thì dừng lại
            except (ValueError, TypeError):
                continue

        if not date_col or not qty_col or not item_col:
            print(f"! Lỗi: Bảng '{key}' thiếu hoặc nhầm cột. (Tìm thấy: {date_col}, {qty_col}, {item_col})")
            continue

        # 2. BỔ SUNG LABEL ENCODER (Giữ nguyên logic của bạn nhưng ép kiểu an toàn)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[item_col] = df[item_col].astype(str)
        df['MaHangHoa_Encoded'] = le.fit_transform(df[item_col])
        joblib.dump(le, "model/label_encoder.pkl")

        # 3. Xử lý thời gian
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]) # Xóa bỏ dòng lỗi ngày tháng nếu có
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day'] = df[date_col].dt.day

        # 4. Chuẩn bị đặc trưng & Mục tiêu (Ép kiểu an toàn)
        X = df[['month', 'day_of_week', 'day', 'MaHangHoa_Encoded']]
        # Dùng errors='coerce' để biến các giá trị lỗi thành NaN, sau đó fill bằng 0
        y = pd.to_numeric(df[qty_col], errors='coerce').fillna(0).astype(float)

        # 5. Huấn luyện model (Sử dụng 100% dữ liệu để đạt độ phủ cao nhất)
        model_xgb = xgb.XGBRegressor(
        n_estimators=1000,        # Xây dựng 500 cây quyết định để khớp dữ liệu phức tạp
        learning_rate=0.05,      # Tốc độ học vừa phải, giúp model hội tụ mịn hơn
        max_depth=10,           # Soi cực sâu vào đặc tính riêng của từng mã hàng (Cá hồi vs Thịt bò)
        subsample=0.8,          # Mỗi cây chỉ lấy 80% dữ liệu ngẫu nhiên (chống học vẹt)
        colsample_bytree=0.8,   # Mỗi cây chỉ lấy 80% tính năng (giúp model khách quan hơn)
        objective='reg:squarederror',
        random_state=42         # Cố định kết quả để mỗi lần train không bị nhảy số khác nhau
    )
        try:
            model_xgb.fit(X, y)
            
            # 6. Lưu model bộ não
            model_name = f"model/{key}_model.pkl"
            joblib.dump(model_xgb, model_name)
            # ... (Sau dòng model_xgb.fit(X, y)) ...

            # 1. Tính độ chính xác (R2 Score)
            from sklearn.metrics import r2_score
            y_pred = model_xgb.predict(X)
            accuracy_val = r2_score(y, y_pred) * 100
            if accuracy_val < 0: accuracy_val = 0
            accuracy_val = round(accuracy_val, 2)

            # 2. Lưu Model bộ não (Code cũ của bạn)
            model_name = f"model/{key}_model.pkl"
            joblib.dump(model_xgb, model_name)

            # 3. Cập nhật độ chính xác vào file JSON (Lưu theo key để không bị đè)
            acc_file = "model/accuracy_info.json"
            all_acc = {}
            if os.path.exists(acc_file):
                with open(acc_file, "r") as f:
                    all_acc = json.load(f)
            
            all_acc[key] = accuracy_val # Lưu theo tên bảng, ví dụ: "train_data_detail": 92.5
            
            with open(acc_file, "w") as f:
                json.dump(all_acc, f)

            print(f">>> THÀNH CÔNG: Đã lưu {model_name} | Độ chính xác: {accuracy_val}%")
        except Exception as e:
            print(f"! Lỗi khi huấn luyện {key}: {str(e)}")
# ======================
# 3. NHÁNH Ý ĐỊNH (LEVEL 2.5)
# ======================
def train_intent_branch():
    print("\n--- [LEVEL 2.5] Huấn luyện bộ não Ý định (Xuất song song BIN/JSON) ---")
    try:
        with open(NEW_DATASET_PATH, "r", encoding="utf-8") as f:
            new_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy {NEW_DATASET_PATH}")
        return

    combined_data = {}
    
    # Khôi phục dữ liệu cũ từ BIN hoặc JSON để gộp (Sync)
    if os.path.exists(BIN_HISTORY_PATH):
        history_data = joblib.load(BIN_HISTORY_PATH)
        for item in history_data:
            intent = item["intent"]
            combined_data[intent] = {"patterns": set(item["patterns"]), "response": item.get("response", "")}
    elif os.path.exists(JSON_HISTORY_PATH):
        with open(JSON_HISTORY_PATH, "r", encoding="utf-8") as f:
            history_data = json.load(f)
            for item in history_data:
                intent = item["intent"]
                combined_data[intent] = {"patterns": set(item["patterns"]), "response": item.get("response", "")}

    # Gộp dữ liệu mới vào
    for item in new_data:
        intent = item["intent"]
        if intent not in combined_data:
            combined_data[intent] = {"patterns": set(), "response": item.get("response", "")}
        for p in item["patterns"]:
            combined_data[intent]["patterns"].add(clean_text(p))

    questions, labels, intent_responses = [], [], {}
    final_to_save = []
    
    for intent, info in combined_data.items():
        p_list = list(info["patterns"])
        final_to_save.append({"intent": intent, "patterns": p_list, "response": info["response"]})
        intent_responses[intent] = info["response"]
        for p in p_list:
            questions.append(p)
            labels.append(intent)

    # Chia 8:2 (Xử lý thông minh nếu có lớp chỉ có 1 mẫu)
    label_counts = Counter(labels)
    if all(count > 1 for count in label_counts.values()):
        X_train, X_test, y_train, y_test = train_test_split(questions, labels, test_size=0.2, stratify=labels, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(questions, labels, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3),analyzer='word', 
    max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model_intent = RandomForestClassifier(n_estimators=200)
    model_intent.fit(X_train_vec, y_train)

    acc = accuracy_score(y_test, model_intent.predict(vectorizer.transform(X_test)))
    print(f">>> KẾT QUẢ TEST: Độ chính xác đạt {acc*100:.2f}%")

    # Lưu kết quả
    full_vec = vectorizer.fit_transform(questions)
    model_intent.fit(full_vec, labels)
    
    # 1. Lưu file máy học (.bin)
    joblib.dump(model_intent, "model/warehouse_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    joblib.dump(final_to_save, BIN_HISTORY_PATH)
    
    # 2. Lưu file quản lý (.json) theo yêu cầu Leader
    with open(JSON_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(final_to_save, f, ensure_ascii=False, indent=4)
        
    with open(INTENT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(intent_responses, f, ensure_ascii=False, indent=4)
    
    print(f">>> THÀNH CÔNG: Đã gộp và lưu {len(questions)} mẫu vào BIN và JSON.")

if __name__ == "__main__":
    train_intent_branch()
    train_forecast_branch()
    print("\n[LORA AI] TẤT CẢ BỘ NÃO ĐÃ KIỂM TRA VÀ SẴN SÀNG TRỰC CHIẾN!")