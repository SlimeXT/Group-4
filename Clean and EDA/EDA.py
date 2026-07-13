import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Tải dữ liệu đã clean
df = pd.read_csv('FINAL_dataset_32k_cleaned.csv')
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

sns.set_theme(style="whitegrid")

# 2. Xuất biểu đồ phân bổ nhãn
plt.figure(figsize=(10, 6))
label_counts = df[label_cols].sum().sort_values(ascending=False)
sns.barplot(x=label_counts.values, y=label_counts.index, palette='magma')
plt.title('Phân bổ nhãn độc hại trong tập 32k (Vét sạch 100%)')
plt.tight_layout()
plt.savefig('label_distribution.png', dpi=300) # Xuất ảnh 300 DPI
print("Đã xuất: label_distribution.png")

# 3. Xuất biểu đồ độ dài câu
df['word_count'] = df['comment_text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.histplot(df['word_count'], bins=50, color='teal')
plt.title('Phân bổ độ dài bình luận (Word Count)')
plt.axvline(120, color='red', linestyle='--', label='MAX_LEN = 120')
plt.legend()
plt.savefig('text_length_distribution.png', dpi=300) # Xuất ảnh 300 DPI
print("Đã xuất: text_length_distribution.png")

# 4. Xuất ma trận tương quan nhãn
plt.figure(figsize=(8, 6))
corr = df[label_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Ma trận tương quan đồng xuất hiện của các nhãn')
plt.savefig('label_correlation.png', dpi=300) # Xuất ảnh 300 DPI
print("Đã xuất: label_correlation.png")