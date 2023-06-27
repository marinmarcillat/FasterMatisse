# Faster matisse

FasterMatisse is a photogrammetry software specifically dedicated to 3D model reconstruction in underwater environments.
It's based on Ifremer's Matisse3D but replaces openMVG by Colmap for sparse reconstruction.

It requires an Nvidia GPU with cuda installed. 

You can find the latest release [here](https://github.com/marinmarcillat/FasterMatisse/releases). 

## Configuration

Required navigation file is a dim2 format, see [Matisse3D documentation](https://github.com/IfremerUnderwater/Matisse/blob/master/Deploy/help/MatisseHelp_EN.pdf)

Camera config file examples in [camera_examples](camera_examples)

Camera parameter format: fx, fy, cx, cy, k1, k2, p1, p2 (OPENCV model). More info [here](https://github.com/colmap/colmap/blob/master/src/base/camera_models.h).

Download vocabulary tree from https://demuc.de/colmap/


## Old installation way:

Requires anaconda

Open anaconda console (anaconda powershell prompt)

Installing the conda env:

    cd path/to/fastermatisse
    conda env create -f environment.yml

Launching

    conda activate FasterMatisse
    cd path/to/fastermatisse
    python main.py


