import os
import gc
import pandas as pd
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import logic từ chat.py
from chat import clean_text, get_answer, load_resources, DYNAMIC_RESPONSES

app = Flask(__name__)
CORS(app)

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(BASE_DIR, "data_history", "database_snapshot.bin")

# --- 1. KHỞI TẠO TÀI NGUYÊN KHI START SERVER ---
# Ép buộc nạp model ngay khi khởi động để tránh lỗi 'NoneType'
with app.app_context():
    print(">>> [HỆ THỐNG]: Đang khởi tạo tài nguyên AI...")
    try:
        load_resources()
        print(">>> [HỆ THỐNG]: Nạp tài nguyên thành công!")
    except Exception as e:
        print(f">>> [LỖI]: Không thể nạp tài nguyên: {e}")

@app.route('/')
def home():
    return jsonify({
        "status": "online", 
        "message": "Lora AI Level 3 - Hệ thống đã sẵn sàng!",
        "version": "3.1.0"
    })

# --- ROUTE 1: DỰ BÁO (Tối ưu bộ nhớ) ---
@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    try:
        from predict import LoraPredictor
        predictor = LoraPredictor()
        days_param = request.args.get('days', default=1, type=int)

        if not os.path.exists(SNAPSHOT_PATH):
            return jsonify({"status": "error", "message": "Dữ liệu chưa được đồng bộ. Hãy chạy /api/sync"}), 404

        # Load snapshot và giải phóng ngay
        db_snapshot = joblib.load(SNAPSHOT_PATH)
        df_hanghoa = pd.DataFrame(db_snapshot["tables"]["hanghoa"])
        del db_snapshot 
        
        data, error = predictor.get_forecast_data(df_hanghoa, days_ahead=days_param)
        
        del df_hanghoa
        gc.collect() 

        if error: return jsonify({"status": "error", "message": error}), 500
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi dự báo: {str(e)}"}), 500

# --- ROUTE 2: CHATBOT AI (Bản nâng cấp chống lỗi) ---
@app.route('/api/chat', methods=['GET', 'POST'])
def chat():
    # Import trực tiếp để đảm bảo lấy đúng biến Global từ chat.py
    from chat import vectorizer, model
    
    # Lấy tin nhắn linh hoạt
    if request.method == 'GET':
        user_text = request.args.get('text')
    else:
        data = request.get_json(silent=True)
        user_text = data.get('text') if data else None

    if not user_text:
        return jsonify({"response": "Leader ơi, bạn chưa nhập tin nhắn!", "status": "error"}), 400

    # KIỂM TRA MODEL TRƯỚC KHI CHẠY
    if vectorizer is None or model is None:
        # Nếu chưa load thì thử load lại một lần nữa
        load_resources()
        from chat import vectorizer, model
        if vectorizer is None:
            return jsonify({"response": "Hệ thống AI đang khởi động, vui lòng thử lại sau giây lát!", "status": "error"}), 503

    try:
        cleaned = clean_text(user_text)
        X = vectorizer.transform([cleaned])
        
        intent = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = max(proba)

        # Lấy phản hồi động
        answer = get_answer(intent, user_text, confidence, DYNAMIC_RESPONSES)

        return jsonify({
            "intent": str(intent),
            "confidence": round(float(confidence), 2),
            "response": answer,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi xử lý AI: {str(e)}"}), 500

# --- ROUTE 3: ĐỒNG BỘ DỮ LIỆU ---
@app.route('/api/sync', methods=['GET', 'POST'])
def manual_sync():
    from readDatabase import read_data
    print(">>> [HỆ THỐNG]: Đang đồng bộ dữ liệu mới...")
    if read_data():
        load_resources() # Nạp lại model và JSON mới vào RAM
        gc.collect()
        return jsonify({"status": "success", "message": "Đã cập nhật kiến thức mới từ Cloud!"})
    return jsonify({"status": "error", "message": "Kết nối Database Cloud thất bại."}), 500

if __name__ == "__main__":
    # Trên Render sử dụng biến môi trường PORT
    port = int(os.environ.get("PORT", 10000))
    # Sử dụng threaded=True để xử lý nhiều yêu cầu cùng lúc
    app.run(host='0.0.0.0', port=port, threaded=True)