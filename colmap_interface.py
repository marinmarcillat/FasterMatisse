import os, json
import configparser
import utils
from tqdm import tqdm
from shutil import copy
from PyQt5 import QtCore
import database_add_gps_from_dim2
import colmap_write_kml_from_database
import convert_colmap_poses_to_texrecon_dev
import ext_programs as ep


class ReconstructionThread(QtCore.QThread):
    step = QtCore.pyqtSignal(str)
    prog_val = QtCore.pyqtSignal(int)
    nb_models = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, gui, image_path, project_path, db_path, camera, vocab_tree_path, nav_path, options):
        super(ReconstructionThread, self).__init__()
        self.running = True
        self.get_exec()
        self.gui = gui
        self.image_path = image_path
        self.camera_path = camera
        self.project_path = project_path
        self.vocab_tree_path = vocab_tree_path
        if not os.path.isfile(db_path):
            db_path = os.path.join(project_path, 'main.db')
            self.run_cmd(self.colmap, ep.create_database_command(db_path))
        self.db_path = db_path
        self.models_path = os.path.join(project_path, 'models')
        self.sparse_model_path = os.path.join(project_path, 'sparse')
        self.export_path = os.path.join(project_path, 'export')
        for path in [self.models_path, self.sparse_model_path, self.export_path]:
            if not os.path.isdir(path):
                os.mkdir(path)
        self.dim2_path = nav_path
        self.nav = utils.load_dim2(nav_path)

        self.CPU_features, self.vocab_tree, self.seq, self.spatial, self.refine, self.matching_neighbors, self.texrecon_text, self.skip_reconstruction = options

    def run(self):
        try:
            if not self.skip_reconstruction:
                self.reconstruction()
            self.post_sparse_reconstruction()
            self.meshing()
            self.export_models(self.texrecon_text)

        except RuntimeError:
            self.gui.normalOutputWritten("An error occurred")
        self.prog_val.emit(0)
        self.finished.emit()
        self.running = False

    def end(self):  # sourcery skip: raise-specific-error
        self.prog_val.emit(0)
        self.finished.emit()
        self.running = False
        raise RuntimeError("An error occurred !")

    def get_exec(self):
        self.colmap = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"COLMAP-3.8-windows-cuda\COLMAP.bat")
        self.openMVS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OpenMVS_Windows_x64")
        self.texrecon = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"texrecon\texrecon.exe")

        for path in [self.colmap, self.openMVS, self.texrecon]:
            if not os.path.exists(path):
                self.gui.normalOutputWritten(f"Error: missing executable: {path}")
                self.end()

    def run_cmd(self, prog, args):
        p = ep.command(prog, args, self.gui)
        p.p.waitForFinished(-1)
        if p.error:
            self.end()
        else:
            return 1

    def config_extract_features(self, cpu_features):
        print("Extracting features")
        config_path = os.path.join(self.project_path, 'extract_features.ini')

        config = configparser.ConfigParser()
        config.read(self.camera_path)
        config.add_section('top')
        config.add_section('SiftExtraction')
        config.set('top', 'database_path', self.db_path)
        config.set('top', 'image_path', self.image_path)
        config.set('ImageReader', 'single_camera', str(1))

        if cpu_features:
            config.set('SiftExtraction', 'estimate_affine_shape', str(1))
            config.set('SiftExtraction', 'domain_size_pooling', str(1))

        text1 = '\n'.join(['='.join(item) for item in config.items('top')])
        text2 = '\n'.join(['='.join(item) for item in config.items('ImageReader')])
        text3 = '\n'.join(['='.join(item) for item in config.items('SiftExtraction')])
        text = text1 + '\n[ImageReader]\n' + text2 + '\n[SiftExtraction]\n' + text3
        with open(config_path, 'w') as config_file:
            config_file.write(text)

        return config_path

    def reconstruction(self):
        self.step.emit('extraction')
        config_path = self.config_extract_features(self.CPU_features)
        self.run_cmd(self.colmap, ep.extract_features_command(config_path))

        database_add_gps_from_dim2.add_nav_to_database(self.db_path, self.dim2_path, self.image_path)

        self.step.emit('matching')
        if self.vocab_tree:
            self.run_cmd(self.colmap,
                         ep.match_features_vocab_command(self.vocab_tree_path, self.db_path, self.matching_neighbors))
        if self.seq:
            self.run_cmd(self.colmap, ep.match_features_seq_command(self.db_path, self.matching_neighbors), )
        if self.spatial:
            self.run_cmd(self.colmap, ep.match_features_spatial_command(self.db_path))
        self.run_cmd(self.colmap, ep.match_features_transitive_command(self.db_path))

        self.step.emit('mapping')
        self.run_cmd(self.colmap, ep.hierarchical_mapper_command(self.sparse_model_path, self.db_path, self.image_path))

    def post_sparse_reconstruction(self):
        list_models = next(os.walk(self.sparse_model_path))[1]
        prog = 0
        tot_len = len(list_models)
        for model in tqdm(list_models):
            sparse_model_path = os.path.join(self.sparse_model_path, model)
            dense_model_path = os.path.join(self.models_path, model)
            if not os.path.isdir(dense_model_path):
                os.mkdir(dense_model_path)

            self.prog_val.emit(round((prog / tot_len) * 100))
            prog += 1
            s = f"{str(round(prog / tot_len * 100))} %, {prog} / {tot_len} \r"
            self.nb_models.emit(f'{prog} / {tot_len}')
            self.gui.normalOutputWritten(s)

            self.step.emit('georegistration')
            self.run_cmd(self.colmap, ep.convert_model_command(sparse_model_path))
            self.get_georegistration_file(sparse_model_path)
            self.run_cmd(self.colmap, ep.georegistration_command(sparse_model_path))
            self.run_cmd(self.colmap, ep.convert_model_command(sparse_model_path))
            self.run_cmd(self.colmap, ep.undistort_image_command(self.image_path, sparse_model_path, dense_model_path))
            self.run_cmd(os.path.join(self.openMVS, 'InterfaceCOLMAP.exe'),
                         ep.interface_openmvs_command(dense_model_path))

    def meshing(self):
        list_models = next(os.walk(self.models_path))[1]
        prog = 0
        tot_len = len(list_models)
        for model in tqdm(list_models):
            dense_model_path = os.path.join(self.models_path, model)

            self.gui.set_prog(round((prog / tot_len) * 100))
            prog += 1
            s = f"{str(round(prog / tot_len * 100))} %, {prog} / {tot_len} \r"
            self.nb_models.emit(f'{prog} / {tot_len}')
            self.gui.normalOutputWritten(s)

            self.step.emit('dense')
            self.run_cmd(os.path.join(self.openMVS, 'DensifyPointCloud.exe'),
                         ep.dense_reconstruction_command(dense_model_path))

            self.step.emit('mesh')
            self.run_cmd(os.path.join(self.openMVS, 'ReconstructMesh.exe'),
                         ep.mesh_reconstruction_command(dense_model_path))

            if self.refine:
                self.step.emit('refinement')
                self.gui.normalOutputWritten("Not available yet \r")

            self.step.emit('texture')
            if self.texrecon_text:
                convert_colmap_poses_to_texrecon_dev.colmap2texrecon(os.path.join(dense_model_path, "sparse"),
                                                                     os.path.join(dense_model_path, "images"))
                self.run_cmd(self.texrecon, ep.texrecon_texturing_command(dense_model_path))
            else:
                self.run_cmd(os.path.join(self.openMVS, 'TextureMesh.exe'),
                             ep.openmvs_texturing_command(dense_model_path))

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

    def export_models(self, obj=True):
        list_models = next(os.walk(self.models_path))[1]
        for model in tqdm(list_models):
            files2copy = [os.path.join(self.sparse_model_path, model, "reference_position.txt")]
            model_dir = os.path.join(self.models_path, model)
            if obj:
                model_name = 'textured_mesh.obj'
                files2copy.extend([os.path.join(model_dir, model_name),
                                   os.path.join(model_dir, 'textured_mesh.mtl')])
                i = 0
                while os.path.exists(os.path.join(model_dir, f'textured_mesh_material{str(i).zfill(4)}_map_Kd.png')):
                    files2copy.append(os.path.join(model_dir, f'textured_mesh_material{str(i).zfill(4)}_map_Kd.png'))
                    i += 1
            else:
                model_name = 'textured_mesh.ply'
                files2copy.append(os.path.join(model_dir, model_name))

            model_export_path = os.path.join(self.export_path, model)
            if not os.path.exists(model_export_path):
                os.mkdir(model_export_path)

            for file in files2copy:
                copy(file, os.path.join(model_export_path, os.path.basename(file)))

            list_poses = utils.read_images_text(os.path.join(self.sparse_model_path, model, "images.txt"), [0, 0, 0])
            camera = utils.read_cameras_text(os.path.join(self.sparse_model_path, model, "cameras.txt"))
            sfm = utils.listposes2sfm(list_poses, camera)
            with open(os.path.join(model_export_path, "sfm_data_temp.json"), 'w') as fp:
                json.dump(sfm, fp, sort_keys=True, indent=4)

            lat, long, alt = colmap_write_kml_from_database.extract_ref_nav_from_database(self.db_path)
            colmap_write_kml_from_database.write_kml_file(os.path.join(model_export_path, 'textured_mesh.kml'),
                                                          model_name, lat, long, alt)
