import subprocess
import os
import configparser
import utils
from tqdm import tqdm
from shutil import copy
from PyQt5 import QtCore
import database_add_gps_from_dim2
import colmap_write_kml_from_database
import convert_colmap_poses_to_texrecon_dev
import json


def run_cmd(cmd):
    p = subprocess.run(cmd, check=True)


class ReconstructionThread(QtCore.QThread):
    step = QtCore.pyqtSignal(str)
    prog_val = QtCore.pyqtSignal(int)
    nb_models = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, image_path, project_path, colmap_path, openMVS_path, db_path, camera, vocab_tree_path,
                 nav_path, options):
        super(ReconstructionThread, self).__init__()
        self.running = True
        self.CPU_features, self.vocab_tree, self.seq, self.spatial, self.refine, self.matching_neighbors, self.skip_reconstruction = options
        self.R = Reconstruction(image_path, project_path, colmap_path, openMVS_path, db_path, camera, vocab_tree_path,
                                nav_path)

    def run(self):
        if not self.skip_reconstruction:
            self.R.sparse_reconstruction(self.matching_neighbors, self.CPU_features, self.vocab_tree,
                                         self.seq, self.spatial, self)
        self.R.post_sparse_reconstruction(self)
        self.R.meshing(self.refine, self)
        self.prog_val.emit(0)
        self.finished.emit()
        self.running = False


class Reconstruction:
    def __init__(self, image_path, project_path, colmap_path, openMVS_path, db_path, camera_path, vocab_tree_path,
                 nav_path):
        self.ref_position = []
        self.list_models = None
        self.openMVS = openMVS_path
        self.image_path = image_path
        self.camera_path = camera_path
        self.vocab_tree_path = vocab_tree_path
        self.colmap = os.path.join(colmap_path, 'COLMAP.bat')
        self.project_path = project_path
        if not os.path.isfile(db_path):
            db_path = os.path.join(project_path, 'main.db')
            subprocess.run([self.colmap, "database_creator", "--database_path", db_path])
        self.db_path = db_path
        self.sparse_model_path = os.path.join(project_path, 'sparse')
        if not os.path.isdir(self.sparse_model_path):
            os.mkdir(self.sparse_model_path)
        self.models_path = os.path.join(project_path, 'models')
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)
        if os.path.isfile(nav_path):
            self.dim2_path = nav_path
            self.nav = utils.load_dim2(nav_path)

    def extract_features(self, cpu_features):
        print("Extracting features")
        config_path = os.path.join(self.project_path, 'extract_features.ini')

        config = configparser.ConfigParser()
        config.read(self.camera_path)
        config.add_section('top')
        config.add_section('SiftExtraction')
        config.set('top', 'database_path', self.db_path)
        config.set('top', 'image_path', self.image_path)
        config.set('ImageReader', 'single_camera', str(1))
        # config.set('SiftExtraction', 'edge_threshold', str(20))
        # config.set('SiftExtraction', 'peak_threshold', str(0.0033))

        if cpu_features:
            config.set('SiftExtraction', 'estimate_affine_shape', str(1))
            config.set('SiftExtraction', 'domain_size_pooling', str(1))

        text1 = '\n'.join(['='.join(item) for item in config.items('top')])
        text2 = '\n'.join(['='.join(item) for item in config.items('ImageReader')])
        text3 = '\n'.join(['='.join(item) for item in config.items('SiftExtraction')])
        text = text1 + '\n[ImageReader]\n' + text2 + '\n[SiftExtraction]\n' + text3
        with open(config_path, 'w') as config_file:
            config_file.write(text)

        command = [
            self.colmap, "feature_extractor",
            "--project_path", config_path
        ]
        run_cmd(command)

    def match_features(self, num_nearest_neighbors=10, vocab=True, seq=True, spatial=True):
        print("Matching features...")
        if vocab:
            print(" Vocabulary tree matching")
            command = [
                self.colmap, "vocab_tree_matcher",
                "--VocabTreeMatching.vocab_tree_path", self.vocab_tree_path,
                "--database_path", self.db_path,
                "--VocabTreeMatching.num_nearest_neighbors", str(num_nearest_neighbors),
                "--SiftMatching.guided_matching", str(1),
            ]
            run_cmd(command)
        if seq:
            print(" Sequential matching")
            command = [
                self.colmap, "sequential_matcher",
                "--database_path", self.db_path,
                "--SequentialMatching.overlap", str(num_nearest_neighbors),
                "--SiftMatching.guided_matching", str(1),
            ]
            run_cmd(command)
        if spatial:
            print(" Spatial matching")
            command = [
                self.colmap, "spatial_matcher",
                "--database_path", self.db_path,
                "--SiftMatching.min_inlier_ratio", str(0.2),
                "--SpatialMatching.ignore_z", str(0),
                "--SpatialMatching.max_distance", str(10),
                "--SpatialMatching.max_num_neighbors", str(64),
                "--SiftMatching.guided_matching", str(1),
            ]
            run_cmd(command)
        print(" Transitive matching")
        command = [
            self.colmap, "transitive_matcher",
            "--database_path", self.db_path
        ]
        run_cmd(command)

    def hierarchical_mapper(self):
        print("Hierarchical mapping...")
        command = [
            self.colmap, "hierarchical_mapper",
            "--output_path", self.sparse_model_path,
            "--database_path", self.db_path,
            "--image_path", self.image_path,
        ]
        run_cmd(command)

    def model_aligner(self, model_path):
        command = [
            self.colmap, "model_sfm_gps_aligner",
            "--input_path", model_path,
            "--output_path", model_path,
            "--database_path", self.db_path,
            "--ref_is_gps", str(1),
            "--alignment_type,", "enu",
        ]
        run_cmd(command)

    def geo_registration(self, model_path):
        if not os.path.isfile(os.path.join(model_path, 'georegist.txt')):
            print("No georegistr !")
            return 0
        command = [
            self.colmap, "model_aligner",
            "--input_path", model_path,
            "--ref_images_path", os.path.join(model_path, 'georegist.txt'),
            "--output_path", model_path,
            "--ref_is_gps", str(1),
            "--alignment_type", 'enu',
            "--robust_alignment", str(1),
            "--robust_alignment_max_error", str(3.0)
        ]
        run_cmd(command)

    def get_georegistration_file(self, model_path):
        filename = os.path.join(model_path, 'images.txt')
        img_list = []
        with open(filename) as file:
            img_list.extend(
                line.rstrip("\n").split(' ')[9]
                for n, line in enumerate(file, start=1)
                if n % 2 != 0 and n > 4
            )
        nav_filtered = self.nav[self.nav['file'].isin(img_list)]
        nav_filtered.to_csv(os.path.join(model_path, 'georegist.txt'), index=None, header=None,
                            sep=' ')
        ref_position = [nav_filtered['lat'].iloc[0], nav_filtered['long'].iloc[0], nav_filtered['depth'].iloc[0]]
        with open(os.path.join(model_path, 'reference_position.txt'), 'w') as f:
            f.write(str(ref_position))

    def convert_model(self, model_path, output_type='TXT'):
        command = [
            self.colmap, "model_converter",
            "--input_path", model_path,
            "--output_path", model_path,
            "--output_type", output_type,
        ]
        run_cmd(command)

    def merge_model(self, model1_name, model2_name, combined_path):
        command = [
            self.colmap, "model_merger",
            "--input_path1", os.path.join(self.sparse_model_path, model1_name),
            "--input_path2", os.path.join(self.sparse_model_path, model2_name),
            "--output_path", combined_path,
        ]
        run_cmd(command)

    def undistort_images(self, model_path, output_path):
        command = [
            self.colmap, "image_undistorter",
            "--image_path", self.image_path,
            "--input_path", model_path,
            "--output_path", output_path,
            "--output_type", "COLMAP"
        ]
        run_cmd(command)
        copy(os.path.join(model_path, 'reference_position.txt'), os.path.join(output_path, 'reference_position.txt'))

    def interface_openMVS(self, model_path):
        command = [
            os.path.join(self.openMVS, 'InterfaceCOLMAP.exe'),
            model_path,
            "-w", model_path,
            "-o", os.path.join(model_path, "model.mvs"),
            "--image-folder", os.path.join(model_path, "images"),

        ]
        run_cmd(command)

    def dense_reconstruction(self, model_path):
        command = [
            os.path.join(self.openMVS, 'DensifyPointCloud.exe'),
            "-i", os.path.join(model_path, "model.mvs"),
            "-o", os.path.join(model_path, "dense.mvs"),
            "-w", model_path
        ]
        run_cmd(command)

    def mesh_reconstruction(self, model_path):
        command = [
            os.path.join(self.openMVS, 'ReconstructMesh.exe'),
            "-i", os.path.join(model_path, "dense.mvs"),
            "-o", os.path.join(model_path, "mesh.mvs"),
            "-w", model_path,
            "--constant-weight", str(0),
            "-f", str(1),
        ]
        run_cmd(command)

    def convert2texrecon(self, model_path):
        convert_colmap_poses_to_texrecon_dev.colmap2texrecon(os.path.join(model_path, "sparse"), os.path.join(model_path, "images"))

    def texrecon_texturing(self, model_path):
        command = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), r"texrecon\texrecon"),
            "--keep_unseen_faces",
            os.path.join(model_path, "images"),
            os.path.join(model_path, "mesh.ply"),
            os.path.join(model_path, "textured_mesh")
        ]
        run_cmd(command)



    def refine_mesh(self, model_path):
        command = [
            os.path.join(self.openMVS, 'RefineMesh.exe'),
            "-i", os.path.join(model_path, "mesh.mvs"),
            "-o", os.path.join(model_path, "mesh_refined.mvs"),
            "-w", model_path
        ]
        run_cmd(command)

    def texture_mesh(self, model_path):
        mesh_path = os.path.join(model_path, "mesh_refined.mvs")
        if not os.path.isfile(mesh_path):
            mesh_path = os.path.join(model_path, "mesh.mvs")
        command = [
            os.path.join(self.openMVS, 'TextureMesh.exe'),
            "-i", mesh_path,
            "-o", os.path.join(model_path, "textured_mesh.mvs"),
            "-w", model_path
        ]
        run_cmd(command)

    def convert_mesh(self, model_path):
        command = [
            os.path.join(self.openMVS, 'Viewer.exe'),
            "-i", os.path.join(model_path, "textured_mesh.mvs"),
            "-o", os.path.join(model_path, "textured_mesh.obj"),
            "-w", model_path,
            "--export-type", 'obj'
        ]
        run_cmd(command)

    def get_images_poses(self, list_offset):
        list_models = next(os.walk(self.sparse_model_path))[1]
        list_poses = {}
        for id, model in tqdm(enumerate(list_models)):
            list_poses |= utils.read_images_text(os.path.join(self.sparse_model_path, model, "images.txt"),
                                                 list_offset[id])
        camera = utils.read_cameras_text(os.path.join(self.sparse_model_path, list_models[0], "cameras.txt"))

        return list_poses, camera

    def group_models(self, list_models):
        model_ref = list_models[0]
        pos_ref = utils.read_reference(os.path.join(self.models_path, model_ref, "reference_position.txt"))
        model_list = [os.path.join(self.models_path, model_ref, "textured_mesh.ply")]
        offset_list = [(0, 0, 0)]
        for model in tqdm(list_models[1:]):
            model_list.append(os.path.join(self.models_path, model, "textured_mesh.ply"))

            pos = utils.read_reference(os.path.join(self.models_path, model, "reference_position.txt"))
            offset = utils.get_offset(pos, pos_ref)
            offset_list.append(offset)
        import CC_utils
        CC_utils.merge_models(model_list, offset_list, os.path.join(self.project_path, "export_merged.ply"))
        copy(os.path.join(self.models_path, model_ref, "reference_position.txt"),
             os.path.join(self.project_path, "reference_position.txt"))
        return offset_list

    def sparse_reconstruction(self, param_feature_matching, cpu_features, vc, seq, spatial, thread=None):
        if thread is not None:
            thread.step.emit('extraction')
        self.extract_features(cpu_features)
        database_add_gps_from_dim2.add_nav_to_database(self.db_path, self.dim2_path, self.image_path)
        if thread is not None:
            thread.step.emit('matching')
        self.match_features(param_feature_matching, vc, seq, spatial)
        if thread is not None:
            thread.step.emit('mapping')
        self.hierarchical_mapper()

    def post_sparse_reconstruction(self, thread=None):
        list_models = next(os.walk(self.sparse_model_path))[1]
        prog = 0
        tot_len = len(list_models)
        for model in tqdm(list_models):
            sparse_model_path = os.path.join(self.sparse_model_path, model)
            dense_model_path = os.path.join(self.models_path, model)
            if not os.path.isdir(dense_model_path):
                os.mkdir(dense_model_path)

            if thread is not None:
                thread.prog_val.emit(round((prog / tot_len) * 100))
            prog += 1
            s = f"{str(round(prog / tot_len * 100))} %, {prog} / {tot_len}"
            if thread is not None:
                thread.nb_models.emit(f'{prog} / {tot_len}')
            print(s, end="\r")

            if thread is not None:
                thread.step.emit('georegistration')
            self.convert_model(sparse_model_path)
            self.get_georegistration_file(sparse_model_path)
            self.geo_registration(sparse_model_path)
            # self.model_aligner(sparse_model_path)
            self.convert_model(sparse_model_path)
            self.undistort_images(sparse_model_path, dense_model_path)
            self.interface_openMVS(dense_model_path)

    def meshing(self, refine=True, thread=None):
        list_models = next(os.walk(self.models_path))[1]
        prog = 0
        tot_len = len(list_models)
        for model in tqdm(list_models):
            dense_model_path = os.path.join(self.models_path, model)

            if thread is not None:
                thread.prog_val.emit(round((prog / tot_len) * 100))
            prog += 1
            s = f"{str(round(prog / tot_len * 100))} %, {prog} / {tot_len}"
            if thread is not None:
                thread.nb_models.emit(f'{prog} / {tot_len}')
            print(s, end="\r")

            if thread is not None:
                thread.step.emit('dense')
            self.dense_reconstruction(dense_model_path)
            if thread is not None:
                thread.step.emit('mesh')
            self.mesh_reconstruction(dense_model_path)
            if refine:
                if thread is not None:
                    thread.step.emit('refinement')
                self.refine_mesh(dense_model_path)
            if thread is not None:
                thread.step.emit('texture')
            self.convert2texrecon(dense_model_path)
            self.texrecon_texturing(dense_model_path)

        """if thread is not None:
            thread.step.emit('merge')
        if len(list_models) != 1:
            offset_list = self.group_models(list_models)
        else:
            copy(os.path.join(self.models_path, list_models[0], "textured_mesh.ply"),
                 os.path.join(self.project_path, "export_merged.ply"))
            copy(os.path.join(self.models_path, list_models[0], "textured_mesh.png"),
                 os.path.join(self.project_path, "textured_mesh.png"))
            copy(os.path.join(self.models_path, list_models[0], "reference_position.txt"),
                 os.path.join(self.project_path, "reference_position.txt"))
            offset_list = [[0, 0, 0]]

        list_poses, cameras = self.get_images_poses(offset_list)
        sfm = utils.listposes2sfm(list_poses, cameras)
        with open(os.path.join(self.project_path, "sfm_data_temp.json"), 'w') as fp:
            json.dump(sfm, fp, sort_keys=True, indent=4)

        lat, long, alt = utils.read_reference(os.path.join(self.project_path, "reference_position.txt"))
        colmap_write_kml_from_database.write_kml_file('export_merged.kml', 'export_merged.ply', lat, long, alt)"""
