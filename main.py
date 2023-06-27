import sys, os
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import (
    QDialog, QMainWindow, QFileDialog, QProgressBar
)
from colmap_interface import ReconstructionThread
from main_ui import Ui_MainWindow


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        with open(r'UI/DarkOrange.qss', 'r', encoding='utf-8') as file:
            qss = file.read()
        self.setStyleSheet(qss)

        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QtGui.QIcon('Logo-Ifremer.ico'))
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)

        self.connectActions()

        self.active_pb = self.main_pb

        self.reconstruction_thread = None

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        cursor = self.log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def connectActions(self):
        self.project_directory_B.clicked.connect(lambda: self.selectDir(self.project_directory))
        self.image_dir_B.clicked.connect(lambda: self.selectDir(self.image_dir))
        self.database_B.clicked.connect(lambda: self.selectFile(self.database, "*.db"))
        self.nav_file_B.clicked.connect(lambda: self.selectFile(self.nav_file, "*.dim2"))
        self.camera_config_B.clicked.connect(lambda: self.selectFile(self.camera_config, "*.ini"))
        self.vocab_tree_B.clicked.connect(lambda: self.selectFile(self.vocab_tree, "*.bin"))

        self.launch.clicked.connect(lambda: self.launch_reconstruction())

    def selectFile(self, line, fileType):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        file_path = QFileDialog.getOpenFileName(self, "Open file", "", fileType, options=options)
        if file_path[0] != "":
            line.setText(file_path[0])

    def selectDir(self, line):
        if dir_path := QFileDialog.getExistingDirectory(None, 'Open Dir', r""):
            line.setText(dir_path)

    def set_nb_models(self, val):
        self.model_nb.setText(val)

    def set_prog(self, val):
        self.active_pb.setValue(val)

    def set_step(self, step):
        label_dict = {
            'georegistration': self.georeferencing,
            'extraction': self.feature_extraction,
            'matching': self.feauture_matching,
            'mapping': self.sparse_reconstruction,
            'dense': self.dense_reconstruction,
            'mesh': self.mesh_reconstruction,
            'refinement': self.mesh_refinement,
            'texture': self.mesh_texturing,
        }
        for key in label_dict:
            label_dict[key].setStyleSheet("QLabel {color : white; font-weight: roman}")
        label_dict[step].setStyleSheet("QLabel {color : green; font-weight: bold}")

    def launch_reconstruction(self):
        image_path = self.image_dir.text()
        project_path = self.project_directory.text()
        db_path = self.database.text()
        nav_path = self.nav_file.text()
        vocab_tree_path = self.vocab_tree.text()
        camera = self.camera_config.text()

        CPU_features = self.CPU_features.isChecked()
        vocab_tree = self.vocab_tree_cb.isChecked()
        seq = self.sequential_cb.isChecked()
        spatial = self.spatial_cb.isChecked()
        refine = self.refine.isChecked()
        matching_neighbors = int(self.num_neighbors.text())
        skip_reconstruction = self.skip_reconstruction.isChecked()
        two_view = self.two_view.isChecked()
        img_scaling = int(self.img_scaling.value())
        decimation = float(self.decimation.value())
        options = [CPU_features, vocab_tree, seq, spatial, refine, matching_neighbors, two_view, img_scaling, decimation, skip_reconstruction]

        self.reconstruction_thread = ReconstructionThread(self, image_path, project_path, db_path, camera,
                                                          vocab_tree_path, nav_path, options)
        self.reconstruction_thread.prog_val.connect(self.set_prog)
        self.reconstruction_thread.step.connect(self.set_step)
        self.reconstruction_thread.nb_models.connect(self.set_nb_models)
        self.reconstruction_thread.finished.connect(self.end_reconstruction)
        self.reconstruction_thread.start()

    def end_reconstruction(self):
        label_dict = {
            'georegistration': self.georeferencing,
            'extraction': self.feature_extraction,
            'matching': self.feauture_matching,
            'mapping': self.sparse_reconstruction,
            'dense': self.dense_reconstruction,
            'mesh': self.mesh_reconstruction,
            'refinement': self.mesh_refinement,
            'texture': self.mesh_texturing,
        }

        for value in label_dict.values():
            value.setStyleSheet("QLabel {color : white; font-weight: roman}")

        self.set_prog(0)
        self.normalOutputWritten("Reconstruction ended \r")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
