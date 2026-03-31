import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ======================================================
# 1. CÁC CÂU LỆNH SQL CHIẾN LƯỢC (12 TRUY VẤN NÂNG CAO)
# ======================================================
LIST_QUERIES = {
    # Truy vấn chủ đạo cho AI Training (Dự báo Nhập)
    "train_data_detail": """
        SELECT CAST(pn.ThoiGian AS DATE) AS Ngay, ct.MaHangHoa, SUM(ct.SoLuongThucTeNhap) AS TongSL
        FROM phieunhap pn JOIN chitietphieunhap ct ON pn.MaPhieuNhap = ct.MaPhieuNhap
        GROUP BY Ngay, ct.MaHangHoa ORDER BY Ngay ASC
    """,
    
    # Dự báo Xuất (Nhu cầu thị trường)
    "export_forecast_data": """
        SELECT CAST(ThoiGian AS DATE) AS Ngay, ct.MaHangHoa, SUM(ct.SoLuongThucTeXuat) AS TongSL
        FROM phieuxuat px JOIN chitietphieuxuat ct ON px.MaPhieuXuat = ct.MaPhieuXuat
        GROUP BY Ngay, ct.MaHangHoa ORDER BY Ngay ASC
    """,

    # --- Các truy vấn quản trị và cảnh báo ---
    "expiry_warning": "SELECT MaHangHoa, SUM(SoLuongTon) as Ton, DATEDIFF(MIN(HanSuDung), CURDATE()) as NgayConLai FROM tonkho_chitiet WHERE TrangThai = 1 GROUP BY MaHangHoa HAVING NgayConLai < 30",
    "shelf_efficiency": "SELECT MaLoaiKho, SUM(DaChua) as DaChua, SUM(TongSucChua) as TongTS, ROUND((SUM(DaChua)/SUM(TongSucChua))*100, 2) as PhanTram FROM kehanghoa GROUP BY MaLoaiKho",
    "low_stock_trigger": "SELECT MaHangHoa, DaChua, MucToiThieuCanhBao FROM kehanghoa WHERE DaChua < MucToiThieuCanhBao",
    
    # Top/Bottom hàng hóa trong ngày
    "top_import_today": "SELECT ct.MaHangHoa, SUM(ct.SoLuongThucTeNhap) as SL FROM chitietphieunhap ct JOIN phieunhap pn ON ct.MaPhieuNhap = pn.MaPhieuNhap WHERE CAST(pn.ThoiGian AS DATE) = (SELECT MAX(CAST(ThoiGian AS DATE)) FROM phieunhap) GROUP BY ct.MaHangHoa ORDER BY SL DESC LIMIT 5",
    "top_export_today": "SELECT ct.MaHangHoa, SUM(ct.SoLuongThucTeXuat) as SL FROM chitietphieuxuat ct JOIN phieuxuat px ON ct.MaPhieuXuat = px.MaPhieuXuat WHERE CAST(px.ThoiGian AS DATE) = (SELECT MAX(CAST(ThoiGian AS DATE)) FROM phieuxuat) GROUP BY ct.MaHangHoa ORDER BY SL DESC LIMIT 5",

    # Chiến lược Tồn kho an toàn (Safety Stock)
    "safety_stock_analysis": """
        SELECT MaHangHoa, AVG(TongSL_Xuat) AS AvgUsage, MAX(TongSL_Xuat) AS MaxUsage,
        ROUND((MAX(TongSL_Xuat) * 7) - (AVG(TongSL_Xuat) * 3), 0) AS SafetyStock_GoiY
        FROM (
            SELECT CAST(px.ThoiGian AS DATE) as Ngay, ct.MaHangHoa, SUM(ct.SoLuongThucTeXuat) AS TongSL_Xuat
            FROM phieuxuat px JOIN chitietphieuxuat ct ON px.MaPhieuXuat = ct.MaPhieuXuat GROUP BY Ngay, ct.MaHangHoa
        ) AS History GROUP BY MaHangHoa
    """
}

# ======================================================
# 2. CÔNG CỤ XỬ LÝ DỮ LIỆU AI (LOGIC NÂNG CẤP)
# ======================================================

def enrich_features_for_ai(df, label_encoder_path='data_history/le_hanghoa.bin'):
    """ 
    Hàm xử lý DataFrame thô từ SQL thành dữ liệu sẵn sàng cho XGBoost 
    """
    if df.empty: return df
    
    # 1. Chuẩn hóa thời gian
    df['Ngay'] = pd.to_datetime(df['Ngay'])
    
    # 2. Xử lý tính liên tục (Fill missing dates với SL = 0)
    all_items = df['MaHangHoa'].unique()
    full_range = pd.date_range(df['Ngay'].min(), df['Ngay'].max())
    
    list_resampled = []
    for item in all_items:
        temp = df[df['MaHangHoa'] == item].set_index('Ngay')
        temp = temp.reindex(full_range).fillna({'MaHangHoa': item, 'TongSL': 0})
        list_resampled.append(temp)
    
    df = pd.concat(list_resampled).reset_index().rename(columns={'index': 'Ngay'})
    
    # 3. Feature Engineering (Thời gian + Biến trễ)
    df = df.sort_values(['MaHangHoa', 'Ngay'])
    df['Thu'] = df['Ngay'].dt.dayofweek
    df['Thang'] = df['Ngay'].dt.month
    df['Is_Weekend'] = df['Thu'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Lags & Windows
    df['SL_HomQua'] = df.groupby('MaHangHoa')['TongSL'].shift(1)
    df['SL_TuanTruoc'] = df.groupby('MaHangHoa')['TongSL'].shift(7)
    df['TB_3Ngay'] = df.groupby('MaHangHoa')['TongSL'].transform(lambda x: x.rolling(3).mean())
    
    # 4. Mã hóa MaHangHoa
    le = LabelEncoder()
    df['MaHangHoa_ID'] = le.fit_transform(df['MaHangHoa'])
    joblib.dump(le, label_encoder_path) # Lưu để giải mã sau này
    
    # Đánh dấu dữ liệu mới nhất (ví dụ 100 dòng cuối) là 'Realtime'
    df['TrongSo'] = 1.0
    df.iloc[-100:, df.columns.get_loc('TrongSo')] = 10.0 # Ưu tiên gấp 10 lần
    
    return df.fillna(0)

# ======================================================
# 3. HÀM KHỞI TẠO SNAPSHOT
# ======================================================
# ======================================================
# 3. HÀM CẬP NHẬT SNAPSHOT (KHÔNG GHI ĐÈ)
# ======================================================

def update_logic_snapshot(new_queries=None):
    try:
        # --- NÂNG CẤP QUAN TRỌNG: LẤY ĐƯỜNG DẪN TUYỆT ĐỐI ---
        # Lấy thư mục chứa chính file query.py này
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Luôn trỏ vào folder data_history nằm CÙNG CẤP với file script
        target_dir = os.path.join(current_script_dir, 'data_history')
        target_file = os.path.join(target_dir, "database_snapshot.bin")

        # In ra để ông kiểm tra (Debug)
        print(f">>> [DEBUG]: Thư mục mục tiêu: {target_dir}")

        # 1. Khởi tạo cấu trúc mặc định
        db_snapshot = {
            "query_logic": {},
            "ai_metadata": {
                "features": ["MaHangHoa_ID", "Thu", "Thang", "Is_Weekend", "SL_HomQua", "SL_TuanTruoc", "TB_3Ngay"],
                "target": "TongSL",
                "model_type": "XGBoost_Regressor_v1"
            }
        }

        # 2. Kiểm tra và Load file cũ nếu có (Dùng đường dẫn tuyệt đối)
        if os.path.exists(target_file):
            try:
                loaded_data = joblib.load(target_file)
                if isinstance(loaded_data, dict):
                    db_snapshot.update(loaded_data)
                    print(f"🔄 Đã tìm thấy snapshot tại đúng vị trí. Đang hợp nhất...")
            except Exception as e:
                print(f"⚠️ Lỗi đọc file cũ: {e}")
        else:
            # Nếu chưa có folder thì mới tạo
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                print(f"📁 Khởi tạo thư mục mới: {target_dir}")

        # 3. Cập nhật query mới
        if new_queries:
            db_snapshot["query_logic"].update(new_queries)
            print(f"➕ Đã nạp thêm {len(new_queries)} kịch bản truy vấn.")

        # 4. Lưu lại
        joblib.dump(db_snapshot, target_file, compress=3)
        
        print("=" * 60)
        print(f"✅ LORA AI: Tổng cộng có {len(db_snapshot['query_logic'])} lệnh.")
        print(f"📂 Snapshot CHÍNH XÁC tại: {target_file}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Lỗi: {e}")
if __name__ == "__main__":
    # Chạy lần đầu với danh sách mặc định
    update_logic_snapshot(new_queries=LIST_QUERIES)
    
    # VÍ DỤ: Lần sau bạn chỉ cần gọi:
    # update_logic_snapshot(new_queries={"lenh_moi": "SELECT * FROM..."}) 
    # Nó sẽ giữ nguyên 12 lệnh cũ và thêm lệnh thứ 13 vào.