import os
import gc
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import logic (Lazy loading bên trong chat.py giúp import ở đây rất nhẹ)
from chat import clean_text, get_answer, load_resources, resources_loaded
from readDatabase import read_data 

app = Flask(__name__)
CORS(app)

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(BASE_DIR, "data_history", "database_snapshot.bin")

@app.route('/')
def home():
    return jsonify({"status": "online", "message": "Lora AI Level 3 - Ready to work!"})

# --- ROUTE 1: DỰ BÁO ---
@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    try:
        from predict import LoraPredictor
        predictor = LoraPredictor()
        days_param = request.args.get('days', default=1, type=int)

        if not os.path.exists(SNAPSHOT_PATH):
            return jsonify({"status": "error", "message": "Snapshot chưa tồn tại. Chạy /api/sync trước."}), 404

        import joblib
        db_snapshot = joblib.load(SNAPSHOT_PATH)
        df_hanghoa = pd.DataFrame(db_snapshot["tables"]["hanghoa"])
        del db_snapshot 
        
        data, error = predictor.get_forecast_data(df_hanghoa, days_ahead=days_param)
        
        del df_hanghoa
        gc.collect() # Giải phóng RAM ngay

        if error: return jsonify({"status": "error", "message": error}), 500
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- ROUTE 2: CHATBOT AI ---
# Sửa dòng này để nhận cả GET và POST
# --- ROUTE 2: CHATBOT AI (BẢN ĐÃ SỬA) ---
@app.route('/api/chat', methods=['GET', 'POST'])
def chat():
    from chat import vectorizer, model, DYNAMIC_RESPONSES # Đảm bảo các biến này đã sẵn sàng
    
    if request.method == 'GET':
        user_text = request.args.get('text')
    else:
        data = request.get_json(silent=True)
        user_text = data.get('text') if data else None

    if not user_text:
        return jsonify({"response": "Leader chưa nhập tin nhắn nè!"}), 400

    try:
        # 1. Làm sạch văn bản
        cleaned = clean_text(user_text)
        
        # 2. Dự đoán Intent bằng Model AI giống hệt chat.py
        X = vectorizer.transform([cleaned])
        intent = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = max(proba)

        # 3. Lấy câu trả lời từ file JSON (DYNAMIC_RESPONSES)
        answer = get_answer(intent, user_text, confidence, DYNAMIC_RESPONSES)

        # 4. Trả kết quả THẬT về cho Web
        return jsonify({
            "intent": str(intent),
            "confidence": float(confidence),
            "response": answer,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi xử lý AI: {str(e)}"}), 500
# --- ROUTE 3: SYNC DỮ LIỆU (Nút kích hoạt) ---
@app.route('/api/sync', methods=['GET', 'POST'])
def manual_sync():
    if read_data():
        load_resources() # Nạp lại tài nguyên vào RAM sau khi sync
        gc.collect()
        return jsonify({"status": "success", "message": "Đồng bộ dữ liệu Cloud thành công!"})
    return jsonify({"status": "error", "message": "Đồng bộ thất bại. Kiểm tra DB Cloud."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # KHÔNG gọi nạp dữ liệu ở đây để tránh treo Port khi khởi động
    app.run(host='0.0.0.0', port=port)