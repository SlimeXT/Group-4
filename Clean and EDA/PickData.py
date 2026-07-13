import pandas as pd

print("="*60)
print(" CHƯƠNG TRÌNH XÂY DỰNG TẬP DỮ LIỆU LARGE SCALE (~32K MẪU) ")
print("="*60)

print("\n--- [1] Đọc và làm sạch tập dữ liệu gốc ---")
try:
    df = pd.read_csv('train.csv')
    original_len = len(df)
    # Lọc bỏ các dòng bị copy-paste spam trùng lặp nội dung
    df = df.drop_duplicates(subset=['comment_text']).reset_index(drop=True)
    print(f"Đã tải {original_len} mẫu. Sau khi xóa spam trùng lặp còn: {len(df)} mẫu duy nhất.")
except FileNotFoundError:
    print(" LỖI: Không tìm thấy file 'train.csv'.")
    exit()

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("\n--- [2] Vét sạch 100% Dữ liệu Độc hại ---")
# Tạo cột đếm tổng số nhãn độc hại của mỗi câu
df['toxic_sum'] = df[label_cols].sum(axis=1)

# Lọc ra toàn bộ các câu có ít nhất 1 nhãn độc hại
toxic_df = df[df['toxic_sum'] > 0].copy()
n_toxic = len(toxic_df)
print(f"=> Đã gom vét toàn bộ {n_toxic} bình luận ĐỘC HẠI duy nhất từ tập gốc.")

print("\n--- [3] Lấy bình luận SẠCH (Clean) để cân bằng 50:50 ---")
# Lọc nhóm Clean (Không có nhãn độc hại nào)
clean_df = df[df['toxic_sum'] == 0].copy()

# Bốc số lượng Clean BẰNG ĐÚNG số lượng Toxic để tỷ lệ là 1:1
if len(clean_df) >= n_toxic:
    clean_sampled = clean_df.sample(n=n_toxic, random_state=42)
    print(f"=> Đã lấy ngẫu nhiên {n_toxic} bình luận SẠCH.")
else:
    print(" LỖI: Không đủ bình luận sạch!")
    exit()

print("\n--- [4] Gộp và Xáo trộn dữ liệu (Shuffle) ---")
# Xóa bỏ cột đếm phụ trước khi xuất file
toxic_df = toxic_df.drop(columns=['toxic_sum'])
clean_sampled = clean_sampled.drop(columns=['toxic_sum'])

# Gộp 2 tập lại
balanced_df = pd.concat([toxic_df, clean_sampled], ignore_index=True)

# Xáo trộn toàn bộ dữ liệu (rất quan trọng để mô hình không học vẹt)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# LƯU RA FILE MỚI
output_name = 'balanced_dataset_32k.csv'
balanced_df.to_csv(output_name, index=False)
print(f"\n🎉 HOÀN TẤT! Đã lưu tập dữ liệu ({len(balanced_df)} mẫu) vào file '{output_name}'")

# In thống kê cuối cùng
print("\n|=== THỐNG KÊ PHÂN BỔ NHÃN TRÊN TẬP 32K ===|")
for label in label_cols:
    count = balanced_df[label].sum()
    print(f"| - {label.ljust(15)}: {count} mẫu")
print(f"| - Clean          : {len(clean_sampled)} mẫu")
print(f"| TỔNG KÍCH THƯỚC  : {len(balanced_df)} mẫu (Tỷ lệ 50:50)")
print("|==========================================|")