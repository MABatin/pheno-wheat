import random
import shutil
import os
import glob


def read_images(path):
    try:
        images = []
        for image in glob.iglob(f'{path}/**/*', recursive=True):
            # check if the image ends with png
            if (image.endswith(".png")) or (image.endswith(".jpg")):
                images.append(image)
        return images
    except Exception:
        print("Error while reading images")


def copy_image_only(split):
    # path to source directory
    src_dir = f'Wheat/data/SPIKE_main/{split}'

    # path to destination directory
    dest_dir = f'Wheat/data/SPIKE_main_imgonly/{split}'
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    # getting all the files in the source directory
    files = read_images(src_dir)

    for file in files:
        shutil.copy2(file, dest_dir)


def move_images(src, dst, limit):
    if not os.path.isdir(dst):
        os.makedirs(dst)

    folders = glob.glob(f'{src}/*', recursive=True)

    i = 0
    while True:
        folder = random.sample(folders, 1)[0]
        files = read_images(folder)
        file = random.choice(files)
        filename = os.path.basename(file)
        if not os.path.exists(os.path.join(dst, filename)):
            shutil.copy2(file, dst)
            i = i+1
        if i == limit:
            break


def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return len(unique_list)


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

    # return string
    return str1


def check(dir):
    images = read_images(dir)
    file_list = []
    for image in images:
        image = os.path.basename(image)
        filename = image.split('_')[:2]
        filename.insert(1, '_')
        filename = listToString(filename)
        file_list.append(filename)
        # print(filename)
    return unique(file_list)


if __name__ == '__main__':
    splits = [('train', 3200), ('test', 400), ('val', 400)]
    for (split, limit) in splits:
        print(f'Copying for {split}......')
        src = f'Wheat/practice/cropped/SPIKE_main_1/{split}/mask'
        dst = f'Wheat/practice/cropped/Spikelet_main/{split}'
        move_images(src, dst, limit)
        print(f'Spike images: {check(dst)}')
