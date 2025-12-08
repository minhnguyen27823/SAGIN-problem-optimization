import numpy as np
import pickle
import random
import os
def load_data(file_in):
    if file_in.endswith(".npy"):
        return np.load(file_in, allow_pickle=True).item()
    elif file_in.endswith(".pkl"):
        with open(file_in, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("File phải có đuôi .npy hoặc .pkl")
    
data = load_data("satellite/final_data.npy")

def get_random_300(data, n_steps=300, max_trials=1000, reindex=True):
    
    # data = load_data(file_in)
    all_steps = sorted(data.keys())
    min_step, max_step = min(all_steps), max(all_steps)

    for _ in range(max_trials):
        start_step = random.randint(min_step, max_step - n_steps)
        segment_raw = {t: data.get(t, []) for t in range(start_step, start_step + n_steps)}

        # kiểm tra có đủ dữ liệu và không rỗng
        if all(len(v) > 0 for v in segment_raw.values()):
            if reindex:
                # reset key về 0..n_steps-1
                segment = {i: segment_raw[t] for i, t in enumerate(range(start_step, start_step + n_steps))}
                return segment
            else:
                return segment_raw

    raise RuntimeError(f"❌ Không tìm được đoạn {n_steps} step hợp lệ sau {max_trials} lần thử")
print(os.getcwd())
