import os
import shutil
import random

def get_all_files(folder):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

source_real = "raw_dataset/for-norm/for-norm/training/real"
source_fake = "raw_dataset/for-norm/for-norm/training/fake"

target_real = "dataset/real"
target_fake = "dataset/fake"

os.makedirs(target_real, exist_ok=True)
os.makedirs(target_fake, exist_ok=True)

if not os.path.exists(source_real):
    print("Real path not found:", source_real)
    exit()

if not os.path.exists(source_fake):
    print("Fake path not found:", source_fake)
    exit()

real_list = get_all_files(source_real)
fake_list = get_all_files(source_fake)

print("Total Real files found:", len(real_list))
print("Total Fake files found:", len(fake_list))

if len(real_list) == 0 or len(fake_list) == 0:
    print("No files found. Check dataset structure again!")
    exit()

real_sample_size = min(800, len(real_list))
fake_sample_size = min(800, len(fake_list))

real_files = random.sample(real_list, real_sample_size)
fake_files = random.sample(fake_list, fake_sample_size)

for f in real_files:
    shutil.copy(f, target_real)

for f in fake_files:
    shutil.copy(f, target_fake)

print("Subset created successfully!")
print(f"Copied {real_sample_size} real files and {fake_sample_size} fake files.")