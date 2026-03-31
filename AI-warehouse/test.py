import mysql.connector
import pandas as pd

# 1. Cấu hình kết nối Database
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '', # Điền password database của bạn nếu có
    'database': 'trainai', # Thay bằng tên DB chứa bảng phieuxuat/phieunhap
}

# 2. Câu lệnh SELECT bạn vừa cung cấp
sql_query = """
    SELECT CAST(ThoiGian AS DATE) AS Ngay, ct.MaHangHoa, SUM(ct.SoLuongThucTeXuat) AS TongSL
    FROM phieuxuat px JOIN chitietphieuxuat ct ON px.MaPhieuXuat = ct.MaPhieuXuat
    GROUP BY Ngay, ct.MaHangHoa ORDER BY Ngay ASC
"""

try:
    # 3. Chạy lệnh để lấy dữ liệu THẬT
    conn = mysql.connector.connect(**db_config)
    df = pd.read_sql(sql_query, conn)
    conn.close()

    # 4. Ghi trực tiếp ra file .arff
    with open('data_from_sql.arff', 'w', encoding='utf-8') as f:
        f.write("@RELATION warehouse_export\n\n")
        f.write("@ATTRIBUTE Ngay DATE \"yyyy-MM-dd\"\n")
        f.write("@ATTRIBUTE MaHangHoa STRING\n")
        f.write("@ATTRIBUTE TongSL NUMERIC\n")
        f.write("\n@DATA\n")
        
        # Định dạng ngày tháng cho chuẩn Weka
        df['Ngay'] = pd.to_datetime(df['Ngay']).dt.strftime('%Y-%m-%d')
        
        # Xuất dữ liệu
        df.to_csv(f, header=False, index=False, lineterminator='\n')
    
    print("✅ Đã lấy dữ liệu từ Database và tạo file .arff thành công!")

except Exception as e:
    print(f"❌ Lỗi khi chạy SQL: {e}")