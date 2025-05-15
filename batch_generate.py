import os
import subprocess
import numpy as np

#설정
N_runs = 300

# 저장 폴더 만들기
os.makedirs('gp_data_batches', exist_ok=True)

for seed in range(N_runs):
    print(f"[INFO] Running simulation {seed+1}/{N_runs}")

    # 환경변수로 시드 넘기기
    os.environ['SIM_SEED'] = str(seed)

    # main.py 실행
    subprocess.run(["python", "main_custom.py"])

    # generate_gp_data.py 실행 (데이터 생성)
    subprocess.run(["python", "generate_gp_data.py"])  

    # 파일 옮기기
    os.rename("gp_training_data.npz", f"gp_data_batches/gp_training_data_seed{seed}.npz")

print("Batch Data Generation Completed")
