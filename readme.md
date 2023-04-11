Requires anaconda

Open anaconda console (anaconda powershell prompt)

installing the conda env:

    conda create --name FasterMatisse python=3.10 -y
    conda activate FasterMatisse
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda install "boost=1.74" "cgal=5.4" cmake ffmpeg "gdal=3.5" laszip "matplotlib=3.5" "mysql=8.0" "numpy=1.22" "opencv=4.5" "openmp=8.0" "pcl=1.12" "pdal=2.4" "psutil=5.9" pybind11 "qhull=2020.2" "qt=5.15.4" "scipy=1.8" sphinx_rtd_theme tbb tbb-devel "xerces-c=3.2" geopy tqdm pyqt pandas -y

Launching

    conda activate FasterMatisse
    cd path/to/fastermatisse
    python main.py

Download vocabulary tree from https://demuc.de/colmap/