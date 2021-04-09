import shutil
import os
import random

val_ratio = 0.2

root = 'C:\\Users\\Leonardo Capozzi\\Desktop\\hid\\train'
target = 'C:\\Users\\Leonardo Capozzi\\Desktop\\hid\\val'

folders = []

for subject_id in os.listdir(root):
    folders.append(subject_id)

val_number = int(len(folders)*val_ratio)

folders = random.sample(folders, val_number)

for f in folders:
    print(f)
    shutil.move(os.path.join(root, f), os.path.join(target, f))

