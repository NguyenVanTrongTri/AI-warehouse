import os
import json
import joblib
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timedelta

class LoraPredictor:
    def __init__(self):
        self.model_path = "model/train_data_detail_model.pkl"
        self.encoder_path = "model/label_encoder.pkl"
        self.acc_file = "model/accuracy_info.json"

    def get_forecast_data(self, df_hanghoa, days_ahead=1):
        """
        NÂNG CẤP: Chỉ dự báo cho mặt hàng có is_new = 1 (Dữ liệu Web).
        Dữ liệu cũ (is_new = 0) chỉ đóng vai trò tri thức đã học trong Model.
        """
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.encoder_path):
                return None, "Bộ não dự báo chưa sẵn sàng (Thiếu Model/Encoder)."

            # Load tri thức đã học từ 193k dòng cũ
            forecast_model = joblib.load(self.model_path)
            le = joblib.load(self.encoder_path)
            
            # --- LỌC ĐỐI TƯỢNG DỰ BÁO ---
            # Chỉ lấy các mặt hàng vừa đồng bộ từ Web (is_new=1)
            # Nếu không có cột is_new, mặc định lấy hết (để tránh lỗi code cũ)
            if 'is_new' in df_hanghoa.columns:
                df_target = df_hanghoa[df_hanghoa['is_new'] == 1]
            else:
                df_target = df_hanghoa

            if df_target.empty:
                return None, "Hiện tại không có mặt hàng mới nào từ Web (is_new=1) để dự báo."

            # --- TÍNH TOÁN NGÀY MỤC TIÊU ---
            target_date = datetime.now() + timedelta(days=int(days_ahead))
            features = {
                'month': target_date.month,
                'day_of_week': target_date.weekday(),
                'day': target_date.day
            }
            
            prediction_results = []
            for _, row in df_target.iterrows():
                ma_hang = str(row['MaHangHoa'])
                try:
                    # Chuyển mã hàng sang số (dựa trên tri thức cũ đã nạp vào Encoder)
                    ma_encoded = le.transform([ma_hang])[0]
                except:
                    # Nếu mã hàng quá mới (Web có nhưng bộ Train cũ chưa thấy)
                    # Chúng ta bỏ qua vì AI chưa đủ kiến thức về mã này
                    continue 

                X_input = pd.DataFrame([{**features, 'MaHangHoa_Encoded': ma_encoded}])
                prediction = forecast_model.predict(X_input)[0]
                qty = max(0, round(float(prediction), 1))
                
                # Ngưỡng tối thiểu để gợi ý nhập hàng
                if qty > 0.5: 
                    prediction_results.append({
                        "ma_hang": row['MaHangHoa'],
                        "ten_hang": row['TenHangHoa'],
                        "dvt": row.get('DonViTinh', 'Cái'),
                        "qty": qty
                    })
            
            # Lấy độ chính xác từ file json (Kết quả của quá trình Train 193k dòng)
            accuracy = "N/A"
            if os.path.exists(self.acc_file):
                with open(self.acc_file, "r") as f:
                    accuracy = json.load(f).get("train_data_detail", "N/A")

            return {
                "date": target_date.strftime('%d/%m/%Y'),
                "accuracy": accuracy,
                "items": prediction_results
            }, None

        except Exception as e:
            return None, f"Lỗi Predict: {str(e)}"

    def format_to_table(self, data):
        """Hiển thị kết quả chỉ dành cho dữ liệu thực tế đang vận hành"""
        if not data or not data['items']: 
            return "LORA AI: Không có mặt hàng Web nào cần nhập thêm vào ngày này."
        
        headers = ["Mã Hàng", "Tên Hàng Hóa", "ĐVT", "Dự Kiến"]
        table_data = [[i['ma_hang'], i['ten_hang'], i['dvt'], i['qty']] for i in data['items']]
        
        output = f"\n🔮 [DỰ BÁO NHU CẦU THỰC TẾ - {data['date']}]\n"
        output += f"📊 Dựa trên tri thức hệ thống (Độ tin cậy: {data['accuracy']}%)\n"
        output += tabulate(table_data, headers=headers, tablefmt="presto")
        output += f"\n(*) Lưu ý: AI chỉ dự báo cho các mặt hàng đang kinh doanh trên Web."
        return output