import numpy as np
import glob

#저장된 데이터 불러오기
file_list = glob.glob('gp_data_batches/gp_training_data_seed*.npz')

Xs = []
Ys = []

print(f"[INFO] Found {len(file_list)} files. Merging...")

for file in sorted(file_list):
    data = np.load(file)
    Xs.append(data['X'])
    Ys.append(data['Y'])

# --- 합치기 ---
X_total = np.concatenate(Xs, axis=0)
Y_total = np.concatenate(Ys, axis=0)

# --- 저장 ---
np.savez('gp_training_data_all.npz', X=X_total, Y=Y_total)

print("Merge Completed!")
print("Final X shape:", X_total.shape, "Final Y shape:", Y_total.shape)
