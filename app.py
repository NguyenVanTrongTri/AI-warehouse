import os
import gc
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import logic
from chat import clean_text, get_answer, load_resources
from predict import LoraPredictor
from readDatabase import read_data 

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(BASE_DIR, "data_history", "database_snapshot.bin")

# Khởi tạo predictor
predictor = LoraPredictor()

# --- HÀM TRỢ GIÚP: ĐẢM BẢO DỮ LIỆU LUÔN MỚI ---
def ensure_latest_data():
    """Gọi read_data để kéo dữ liệu mới nhất từ Aiven về snapshot.bin"""
    print(">>> [HỆ THỐNG]: Đang kiểm tra và cập nhật dữ liệu từ Cloud...")
    return read_data()

# --- ROUTE 1: DỰ BÁO ---
@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    try:
        # TỰ ĐỘNG ĐỒNG BỘ TRƯỚC KHI DỰ BÁO
        ensure_latest_data()

        days_param = request.args.get('days', default=1, type=int)

        if not os.path.exists(SNAPSHOT_PATH):
            return jsonify({"status": "error", "message": "Không thể tạo snapshot dữ liệu."}), 404

        db_snapshot = joblib.load(SNAPSHOT_PATH)
        if "tables" not in db_snapshot or "hanghoa" not in db_snapshot["tables"]:
             return jsonify({"status": "error", "message": "Dữ liệu hàng hóa rỗng."}), 500
             
        df_hanghoa = pd.DataFrame(db_snapshot["tables"]["hanghoa"])
        del db_snapshot 
        
        data, error = predictor.get_forecast_data(df_hanghoa, days_ahead=days_param)
        
        del df_hanghoa
        gc.collect() 

        if error: return jsonify({"status": "error", "message": error}), 500
        return jsonify({"status": "success", "data": data})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi Server: {str(e)}"}), 500

# --- ROUTE 2: CHATBOT AI ---
@app.route('/api/chat', methods=['GET', 'POST'])
def chat():
    # 1. Lấy tin nhắn
    if request.method == 'GET':
        user_text = request.args.get('text')
    else:
        data = request.get_json(silent=True)
        user_text = data.get('text') if data else None

    if not user_text:
        return jsonify({"response": "Leader chưa nhập tin nhắn nè!"}), 400

    try:
        # TỰ ĐỘNG ĐỒNG BỘ ĐỂ AI CÓ KIẾN THỨC MỚI NHẤT
        ensure_latest_data()

        from chat import load_resources, clean_text, vectorizer, model, DYNAMIC_RESPONSES, get_answer
        load_resources() 

        cleaned = clean_text(user_text)
        X = vectorizer.transform([cleaned])
        
        intent = model.predict(X)[0]
        confidence = max(model.predict_proba(X)[0])

        # Trả lời dựa trên dữ liệu vừa sync
        answer = get_answer(intent, user_text, confidence, DYNAMIC_RESPONSES)

        return jsonify({
            "intent": intent,
            "confidence": float(confidence),
            "response": answer,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi xử lý AI: {str(e)}"}), 500

# Giữ lại sync thủ công nếu cần
@app.route('/api/sync', methods=['GET', 'POST'])
def manual_sync():
    if read_data():
        gc.collect()
        return jsonify({"status": "success", "message": "Đã ép buộc đồng bộ Cloud!"})
    return jsonify({"status": "error", "message": "Đồng bộ thất bại."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)