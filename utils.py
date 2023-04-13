import ast, os
import subprocess
import pandas as pd
from geopy.distance import distance
from shutil import copy
from tqdm import tqdm
import numpy as np
import collections
from time import sleep
import copy


def run_cmd(cmd, wait=True):
    pStep = subprocess.Popen(cmd)
    if wait:
        pStep.wait()
        if pStep.returncode != 0:
            print("Warning: step failed")
            print(f'Used command: {cmd}')
            sleep(20)


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


BaseImage = collections.namedtuple(
    "Image", ["id", "rotmat", "tvec", "camera_id", "name"])

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:]))).tolist()
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_images_text(path, offset):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[8])
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8]))) - np.array(offset)
                tvec = tvec.tolist()
                rotmat = qvec2rotmat(qvec).tolist()
                image_name = elems[9]
                elems = fid.readline().split()
                images[image_id] = Image(
                    id=image_id, rotmat=rotmat, tvec=tvec,
                    camera_id=camera_id, name=image_name)
    return images


def listposes2sfm(list_poses, cameras):
    views_template = {
        "key": 0,
        "value": {
            "polymorphic_id": 0,
            "ptr_wrapper": {
                "id": 0,
                "data": {
                    "local_path": "",
                    "filename": "",
                    "width": 0,
                    "height": 0,
                    "id_view": 0,
                    "id_intrinsic": 0,
                    "id_pose": 0
                }
            }
        }
    }
    extrinsics_template = {
            "key": 1,
            "value": {
                "rotation": [
                ],
                "center": []
            }
        }

    width = cameras[1][2]
    height = cameras[1][3]
    fx, fy, cx, cy, k1, k2, k3, k4 = cameras[1][4]

    intrinsics = {
        "key": 0,
        "value": {
            "polymorphic_id": 2147483650,
            "polymorphic_name": "pinhole_radial_k3",
            "ptr_wrapper": {
                "id": 2147485962,
                "data": {
                    "width": width,
                    "height": height,
                    "focal_length": (fx + fy) / 2,
                    "principal_point": [cx, cy],
                    "disto_k3": [k1, k2, k3]
                }
            }
        }
    }

    list_extrinsics = []
    list_views = []
    for pose in list_poses.items():
        _, image = pose
        id, rot, translation, filename = image.id, image.rotmat, image.tvec, image.name
        view = copy.deepcopy(views_template)
        view['key'] = view["value"]["polymorphic_id"] = view["value"]["ptr_wrapper"]["id"] = \
            view["value"]["ptr_wrapper"]["data"]["id_view"] = view["value"]["ptr_wrapper"]["data"]["id_pose"] = id
        view["value"]["ptr_wrapper"]["data"]["filename"] = filename
        view["value"]["ptr_wrapper"]["data"]["width"] = width
        view["value"]["ptr_wrapper"]["data"]["height"] = height
        list_views.append(view)

        ext = copy.deepcopy(extrinsics_template)
        ext["key"] = id
        ext["value"]["rotation"] = rot
        ext["value"]["center"] = translation
        list_extrinsics.append(ext)

    return {
        "sfm_data_version": "0.3",
        "root_path": "",
        "intrinsics": [intrinsics],
        "extrinsics": list_extrinsics,
        "views": list_views,
    }


def set_exifs(image, pos):
    command = [
        'exiftool.exe', f'-EXIF:GPSLongitude={str(pos[1])}', f'-EXIF:GPSLatitude={str(pos[0])}',
        f'-EXIF:GPSAltitude={str(pos[2])}', '-GPSLongitudeRef=West', '-GPSLatitudeRef=North', '-overwrite_original',
        '-q', image
    ]
    run_cmd(command, wait=False)


def set_all_exifs(img_dir, nav):
    print("Setting all exifs")
    img_list = os.listdir(img_dir)
    for img in tqdm(img_list):
        img_path = os.path.join(img_dir, img)
        pos = nav[nav['file'] == img][['lat', 'long', 'depth']].values[0].tolist()
        set_exifs(img_path, pos)
