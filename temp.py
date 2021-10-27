import os
import time
import random
from datetime import datetime
from torchvision.datasets import CelebA


count = 1
downloaded = False
while not downloaded:
    try:
        os.system("rm -rf data/datasets/CelebA/celeba")
        train_set = CelebA(root="data/datasets/CelebA", download=True, split="train", target_type="attr")
        downloaded = True
    except Exception as e:
        print(f"Failed on attempt {count}")
        print(f"{type(e)}: {e}\n")
        count += 1
        sleep_seconds = 3600 + (random.random() * 2 - 1) * 300
        time.sleep(sleep_seconds)

now = datetime.now().strftime("%m/%d/%y %H:%M:%S")
print(f"Succeeded after {count} tries at {now}")
