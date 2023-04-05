import ast, os
import subprocess
import pandas as pd
from geopy.distance import distance
from shutil import copy
from tqdm import tqdm


def run_cmd(cmd, wait = True):
    pStep = subprocess.Popen(cmd)
    if wait:
        pStep.wait()
        if pStep.returncode != 0:
            print("Warning: step failed")
            print(f'Used command: {cmd}')


def load_dim2(dim2_path):
    nav_data = pd.read_csv(dim2_path, sep=";", dtype=str, na_filter=False, header=None)
    nav_data = nav_data.iloc[:, : 13]
    nav_data.columns = ['a', 'date', 'time', 'b', 'c', 'file', 'lat', 'long', 'depth', 'alt', 'yaw', 'pitch', 'roll']
    nav_data = nav_data[["file", "lat", "long", "depth"]]
    nav_data["depth"] = - pd.to_numeric(nav_data["depth"])
    nav_data["lat"] = pd.to_numeric(nav_data["lat"])
    nav_data["long"] = pd.to_numeric(nav_data["long"])
    return nav_data


def read_reference(path):
    with open(path) as f:
        line = f.readlines()
    return ast.literal_eval(line[0])


def get_offset(coords, model_origin):
    offset_z = abs(model_origin[2]) - abs(coords[2])
    offset_x = distance((coords[0], model_origin[1]), (coords[0], coords[1])).m
    if coords[1] < model_origin[1]:
        offset_x = -offset_x
    offset_y = distance((model_origin[0], coords[1]), (coords[0], coords[1])).m
    if coords[0] < model_origin[0]:
        offset_y = -offset_y
    return offset_x, offset_y, offset_z


def merge_models(dir1, dir2, dir_output):
    img_input1 = os.path.join(dir1, 'images.txt')
    img_input2 = os.path.join(dir2, 'images.txt')
    img_output = os.path.join(dir_output, 'images.txt')

    img_input1 = os.path.join(dir1, 'images.txt')
    img_input2 = os.path.join(dir2, 'images.txt')
    img_output = os.path.join(dir_output, 'images.txt')

    copy(img_input1, img_output)
    with open(img_input2) as file:
        lines = [
            line
            for n, line in enumerate(file, start=1)
            if n > 4
        ]
    with open(img_output, "a") as file:
        file.write("".join(lines))


def set_exifs(image, pos):
    command = [
        'exiftool.exe', f'-EXIF:GPSLongitude={str(pos[1])}', f'-EXIF:GPSLatitude={str(pos[0])}',
        f'-EXIF:GPSAltitude={str(pos[2])}', '-GPSLongitudeRef=West', '-GPSLatitudeRef=North', '-overwrite_original', '-q', image
    ]
    run_cmd(command, wait = False)

def set_all_exifs(img_dir, nav):
    print("Setting all exifs")
    img_list = os.listdir(img_dir)
    for img in tqdm(img_list):
        img_path = os.path.join(img_dir, img)
        pos = nav[nav['file'] == img][['lat', 'long', 'depth']].values[0].tolist()
        set_exifs(img_path, pos)
