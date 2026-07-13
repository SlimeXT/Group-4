import pandas as pd
import re

print("="*60)
print(" CHƯƠNG TRÌNH LÀM SẠCH DỮ LIỆU CHUYÊN SÂU (NLP CLEANSING) ")
print("="*60)

# 1. Đọc tập dữ liệu 32k đã cân bằng
input_file = 'balanced_dataset_32k.csv'
try:
    df = pd.read_csv(input_file)
    print(f"Đã tải thành công {len(df)} mẫu từ '{input_file}'")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{input_file}'")
    exit()

# 2. Xây dựng hàm làm sạch chuẩn NLP
def clean_text_advanced(text):
    text = str(text).lower() # Chuyển về chữ thường
    
    # Xóa ký tự xuống dòng, tab
    text = re.sub(r'[\n\t\r]', ' ', text)
    
    # Xóa địa chỉ IP (VD: 192.168.0.1)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', text)
    
    # Xóa URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    
    # Giữ lại chữ cái tiếng Anh, số và dấu cách (Xóa toàn bộ ký tự đặc biệt)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Thu gọn nhiều khoảng trắng liên tiếp thành 1 khoảng trắng duy nhất
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 3. Tiến hành làm sạch
print("\nĐang tiến hành làm sạch văn bản (Quá trình này có thể mất vài giây)...")
df['comment_text'] = df['comment_text'].apply(clean_text_advanced)

# Loại bỏ các dòng mà sau khi làm sạch bị trống (rỗng)
initial_len = len(df)
df = df[df['comment_text'].str.strip() != '']
dropped_rows = initial_len - len(df)
if dropped_rows > 0:
    print(f" Đã loại bỏ {dropped_rows} dòng bị rỗng sau khi làm sạch.")

# 4. Lưu ra file dữ liệu cuối cùng (Final Dataset)
output_file = 'FINAL_dataset_32k_cleaned.csv'
df.to_csv(output_file, index=False)

print(f"\n🎉 HOÀN TẤT! Dữ liệu ĐÃ SẠCH 100% và sẵn sàng đưa vào Deep Learning.")
print(f" Đã lưu file: '{output_file}'")
print("="*60)