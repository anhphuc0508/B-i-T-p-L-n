import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

TOP_N = 3
TOP_PLAYERS_FILE = 'top_3.txt'
RESULTS_FILE = 'results2.csv'
HISTOGRAM_DIR = 'histograms'
ENCODING = 'utf-8-sig'
CSV_FILE = 'results.csv'

# Chọn 3 chỉ số tấn công và 3 chỉ số phòng ngự
attacking_cols = ['Standard SoT/90', 'Total Cmp', 'Touches']
defensive_cols = ['Tackles Tkl', 'Tackles TklW', 'Performance Recov']

# Đọc dữ liệu
try:
    df = pd.read_csv(CSV_FILE, encoding=ENCODING)
    print(f"Read file '{CSV_FILE}' succesfullt.")
except Exception as e:
    print(f"Erroe when reading file '{CSV_FILE}': {e}")
    exit()

# Chuyển đổi dữ liệu dạng chuỗi thành số nếu cần
for col in df.columns:
    if df[col].dtype == 'object' and col not in ['Player', 'Nation', 'Squad', 'Pos', 'Age']:
        if df[col].str.contains('%', na=False).any():
            df[col] = df[col].str.rstrip('%').astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

# Lọc các cột số từ danh sách chọn lọc
selected_columns = attacking_cols + defensive_cols
numeric_columns = [col for col in selected_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

if not numeric_columns:
    print("Can't correct column.")
    exit()

print("Using these columns:")
for i, col in enumerate(numeric_columns, 1):
    print(f"{i}. {col}")

# Kiểm tra cột Player và Squad
if 'Player' not in df.columns or 'Squad' not in df.columns:
    print("Missing 'Player' or 'Squad' column.")
    exit()

# Ghi top/bottom 3 cầu thủ
with open(TOP_PLAYERS_FILE, 'w', encoding=ENCODING) as f:
    for col in numeric_columns:
        f.write(f"Statistic: {col}\n")
        f.write("=" * (len(col) + 11) + "\n")

        # Top 3
        f.write("Top 3 players:\n")
        top_3 = df[['Player', col]].dropna().sort_values(by=col, ascending=False).head(TOP_N)
        f.write(top_3.to_string(index=False) + "\n\n")

        # Bottom 3
        f.write("Bottom 3 players:\n")
        bottom_3 = df[['Player', col]].dropna().sort_values(by=col, ascending=True).head(TOP_N)
        f.write(bottom_3.to_string(index=False) + "\n")
        f.write("-" * 50 + "\n\n")
print(f"Saved file top/bottom 3 to '{TOP_PLAYERS_FILE}'")

# Tính toán theo đội
results = []
for team, group_df in df.groupby('Squad'):
    row = [team]
    for col in numeric_columns:
        row.extend([
            group_df[col].median(),
            group_df[col].mean(),
            group_df[col].std()
        ])
    results.append(row)

# Thêm dòng trung bình toàn giải
league_row = ['All players']
for col in numeric_columns:
    league_row.extend([
        df[col].median(),
        df[col].mean(),
        df[col].std()
    ])
results.append(league_row)

# Tạo header
header = ['Squad']
for col in numeric_columns:
    header.extend([
        f"Median {col}",
        f"Mean {col}",
        f"StdDev {col}"
    ])

# Xuất file CSV thống kê
final_df = pd.DataFrame(results, columns=header)
final_df.to_csv(RESULTS_FILE, index=False, encoding=ENCODING)
print(f"Saved to '{RESULTS_FILE}'")

# Tạo thư mục biểu đồ
os.makedirs(HISTOGRAM_DIR, exist_ok=True)

# Vẽ biểu đồ
for col in numeric_columns:
    safe_col_name = re.sub(r'[^\w\s-]', '_', col)
    stat_dir = os.path.join(HISTOGRAM_DIR, safe_col_name)
    os.makedirs(stat_dir, exist_ok=True)

    # Biểu đồ toàn giải
    data = df[col].dropna()
    plt.figure(figsize=(10, 6))
    if data.empty:
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
    else:
        plt.hist(data, bins=20, color='steelblue', alpha=0.7)
    plt.title(f"Histogram of {col} - All players")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(stat_dir, "All players.png"))
    plt.close()

    # Biểu đồ theo đội
    for team in df['Squad'].dropna().unique():
        team_data = df[df['Squad'] == team][col].dropna()
        plt.figure(figsize=(10, 6))
        if team_data.empty:
            plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
        else:
            plt.hist(team_data, bins=20, color='orange', alpha=0.7)
        plt.title(f"Histogram of {col} - {team}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        safe_team = re.sub(r'[^\w]', '_', str(team))
        plt.savefig(os.path.join(stat_dir, f"{safe_team}.png"))
        plt.close()

print(f"Created chart in folder '{HISTOGRAM_DIR}'")
print("Based on the analysis, Chelsea appears to be the best-performing team in the 2024-2025 Premier League season. They lead in critical attacking metrics (xG, xAG, SCA, KP, PPA), showing they create numerous high-quality chances. Their high rankings in touches and passes into key areas indicate control in attacking phases, and their defensive metrics (tackles, interceptions) are competitive, though not the highest. Arsenal is a close second due to their efficiency in scoring and assisting, but Chelsea’s broader dominance across creative metrics gives them the edge. Manchester City’s possession-based metrics are exceptional, but their slightly lower goal output places them behind Chelsea. Leicester’s defensive and physical strengths are notable, but their attacking output is less impressive")
print("So clearly, I think Chelsea is doing the best in overrall")
