import os
import subprocess
import time
from queue import Queue
import threading

# The combinations to run
combinations = [
    # Temp 0.7
    (0.7, 1.0, 1.05),
    (0.7, 1.0, 1.10),
    (0.7, 1.0, 1.20),
    (0.7, 0.9, 1.00),
    # Temp 0.5
    (0.5, 1.0, 1.00),
    (0.5, 1.0, 1.05),
    (0.5, 1.0, 1.10),
    (0.5, 1.0, 1.20),
    (0.5, 0.9, 1.00),
    (0.5, 0.9, 1.05),
    (0.5, 0.9, 1.10),
    (0.5, 0.9, 1.20)
]

# We have 4 GPUs available: 0, 1, 2, 3
gpus = [0, 1, 2, 3]

# Create a queue of tasks
task_queue = Queue()
for combo in combinations:
    task_queue.put(combo)

def worker(gpu_id):
    while not task_queue.empty():
        try:
            temp, top_p, pen = task_queue.get_nowait()
        except Exception:
            break
            
        print(f"[GPU {gpu_id}] Starting Temp={temp}, Top-p={top_p}, Penalty={pen}")
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONPATH=. python scripts/generate_submission_cli.py --epoch 30 --temperature {temp} --top_p {top_p} --penalty {pen}"
        log_file = f"generate_submission_epoch30_temp{str(temp).replace('.', '')}_pen{str(pen).replace('.', '')}_top{str(top_p).replace('.', '')}.log"
        
        with open(log_file, "w") as f:
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
            
        print(f"[GPU {gpu_id}] Finished Temp={temp}, Top-p={top_p}, Penalty={pen}")
        task_queue.task_done()

# Start workers
threads = []
for gpu in gpus:
    t = threading.Thread(target=worker, args=(gpu,))
    t.start()
    threads.append(t)

# Wait for all tasks to complete
for t in threads:
    t.join()

print("All grid combinations successfully generated!")
