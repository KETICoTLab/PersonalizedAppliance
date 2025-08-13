import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 데이터
data = {
    'Group': ['100k', '100k', '1m', '1m', '32m', '32m'],
    'Type': ['normal', 'lite', 'normal', 'lite', 'normal', 'lite'],
    'Insert Time (s)': [0.35766, 0.02925, 1.48506, 0.09596, 66.95656, 23.14492],
    'Recommend Time (s)': [1.13799, 0.2419, 8.80256, 3.17793, 209.23645, 40.10814],
    'Accuracy': [0.59, 0.59, 0.66, 0.66, 0.82, 0.82]
}
df = pd.DataFrame(data)

# 스타일
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

# Helper 함수
def add_labels(ax, decimals=2):
    for container in ax.containers:
        labels = [f"{bar.get_height():.{decimals}f}" for bar in container]
        ax.bar_label(container, labels=labels, label_type='edge', fontsize=9, padding=2)

# Insert Time
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Group', y='Insert Time (s)', hue='Type', data=df, ax=ax, palette=palette)
ax.set_yscale('log')
ax.set_title('Insert Time')
ax.set_ylabel('Time (s, log)')
ax.set_xlabel("")
add_labels(ax, decimals=2)
plt.tight_layout()
plt.savefig("insert_time_log.png", dpi=300)
plt.savefig("insert_time_log.pdf")
plt.close()

# Recommend Time
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Group', y='Recommend Time (s)', hue='Type', data=df, ax=ax, palette=palette)
ax.set_yscale('log')
ax.set_title('Recommendation Time')
ax.set_ylabel('Time (s, log)')
ax.set_xlabel("")
add_labels(ax, decimals=2)
plt.tight_layout()
plt.savefig("recommend_time_log.png", dpi=300)
plt.savefig("recommend_time_log.pdf")
plt.close()

# Accuracy
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Group', y='Accuracy', hue='Type', data=df, ax=ax, palette=palette)
ax.set_title('Accuracy')
ax.set_ylim(0, 1.0)
ax.set_ylabel('Accuracy')
ax.set_xlabel("")
add_labels(ax, decimals=2)
plt.tight_layout()
plt.savefig("accuracy.png", dpi=300)
plt.savefig("accuracy.pdf")
plt.close()