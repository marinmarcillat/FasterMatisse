import PyInstaller.__main__
import shutil
import os

PyInstaller.__main__.run([
    'main.py',
    '--icon=Logo-Ifremer.ico',
    '--onefile',
], )

paths = ['CloudCompare', 'COLMAP-3.8-windows-cuda', 'OpenMVS_Windows_x64']

#for path in paths:
#    shutil.copytree(path, os.path.join('dist', path))

#shutil.copy('exiftool.exe', os.path.join('dist', 'exiftool.exe'))