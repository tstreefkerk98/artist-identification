import os
import shutil
from tqdm import tqdm
max_size = 300
train_size = 0.8
val_size = 0.9
test_size = 1
save_directory = "C:\\Users\\havel\\Documents\\TUDelft\\22-23\\Q4\\CV\\artist-identification\\wikiart_dataset"
os.mkdir(save_directory)
os.mkdir(os.path.join(save_directory, "train"))
os.mkdir(os.path.join(save_directory, "validation"))
os.mkdir(os.path.join(save_directory, "test"))
directory = "C:\\Users\\havel\\Documents\\TUDelft\\22-23\\Q4\\CV\\artist-identification\\dataset"
for folder in tqdm(os.listdir(directory)):
    if folder == ".gitkeep":
        continue
    artist_name = folder
    full_name_train = os.path.join(save_directory, "train", artist_name)
    os.mkdir(full_name_train)
    full_name_val = os.path.join(save_directory, "validation", artist_name)
    os.mkdir(full_name_val)
    full_name_test = os.path.join(save_directory, "test", artist_name)
    os.mkdir(full_name_test)
    for filename in os.listdir(os.path.join(directory, folder)):
        num = int(filename.split('.')[0])
        src = os.path.join(directory, folder, filename)
        if num < train_size * max_size:
            shutil.copyfile(src, os.path.join(full_name_train, filename))
        elif num < val_size * max_size:
            shutil.copyfile(src, os.path.join(full_name_val, filename))
        else:
            shutil.copyfile(src, os.path.join(full_name_test, filename))

