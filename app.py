import os
import gc
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import logic
from chat import clean_text, get_answer, load_resources, vectorizer, model, DYNAMIC_RESPONSES
from predict import LoraPredictor
from readDatabase import read_data 

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(BASE_DIR, "data_history", "database_snapshot.bin")

# --- 1. KHỞI TẠO TÀI NGUYÊN (CHỈ CHẠY 1 LẦN KHI START SERVER) ---
print(">>> [HỆ THỐNG]: Đang khởi tạo Model AI và Predictor...")
predictor = LoraPredictor()
load_resources() # Load model và vectorizer lên RAM ngay từ đầu

# --- 2. HÀM ĐỒNG BỘ DỮ LIỆU (CHẠY NGẦM HOẶC THỦ CÔNG) ---
def sync_data():
    """Hàm này chỉ nên gọi khi thực sự cần hoặc chạy định kỳ"""
    print(">>> [HỆ THỐNG]: Đang đồng bộ dữ liệu từ Cloud...")
    return read_data()

# --- ROUTE 1: DỰ BÁO ---
@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    try:
        # Thay vì auto-sync mỗi lần, hãy dùng dữ liệu hiện có để phản hồi nhanh
        # Chỉ sync nếu file snapshot không tồn tại
        if not os.path.exists(SNAPSHOT_PATH):
            sync_data()

        days_param = request.args.get('days', default=1, type=int)
        
        db_snapshot = joblib.load(SNAPSHOT_PATH)
        df_hanghoa = pd.DataFrame(db_snapshot["tables"]["hanghoa"])
        del db_snapshot 
        
        data, error = predictor.get_forecast_data(df_hanghoa, days_ahead=days_param)
        
        del df_hanghoa
        gc.collect() 

        if error: return jsonify({"status": "error", "message": error}), 500
        return jsonify({"status": "success", "data": data})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi Server: {str(e)}"}), 500

# --- ROUTE 2: CHATBOT AI (TỐI ƯU TỐC ĐỘ PHẢN HỒI) ---
@app.route('/api/chat', methods=['GET', 'POST'])
def chat():
    # Lấy tin nhắn nhanh
    user_text = request.args.get('text') if request.method == 'GET' else request.get_json(silent=True).get('text')

    if not user_text:
        return jsonify({"response": "Leader chưa nhập tin nhắn kìa!"}), 400

    try:
        # KHÔNG gọi sync_data() ở đây. AI dùng kiến thức từ snapshot hiện tại.
        # KHÔNG gọi load_resources() ở đây. Model đã nằm sẵn trên RAM rồi.

        cleaned = clean_text(user_text)
        X = vectorizer.transform([cleaned])
        
        intent = model.predict(X)[0]
        # Chuyển đổi xác suất để lấy độ tin cậy
        proba = model.predict_proba(X)[0]
        confidence = max(proba)

        # Trả lời dựa trên snapshot hiện tại
        answer = get_answer(intent, user_text, confidence, DYNAMIC_RESPONSES)

        return jsonify({
            "intent": str(intent),
            "confidence": float(confidence),
            "response": answer,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi xử lý AI: {str(e)}"}), 500

# --- ROUTE 3: ĐỒNG BỘ THỦ CÔNG (DÙNG KHI CẦN CẬP NHẬT DỮ LIỆU MỚI) ---
@app.route('/api/sync', methods=['GET', 'POST'])
def manual_sync():
    if sync_data():
        gc.collect()
        return jsonify({"status": "success", "message": "Đã cập nhật dữ liệu mới nhất từ Cloud!"})
    return jsonify({"status": "error", "message": "Đồng bộ thất bại."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)