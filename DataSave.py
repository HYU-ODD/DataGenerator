from config import config_to_trans
from export_utils import *
from carla import ColorConverter as cc

class DataSave:
    def __init__(self, cfg):
        self.cfg = cfg
        self.OUTPUT_FOLDER = None
        self.LIDAR_PATH = None
        self.KITTI_LABEL_PATH = None
        self.CARLA_LABEL_PATH = None
        self.IMAGE_PATH = None
        self.CALIBRATION_PATH = None
        self._generate_path(self.cfg["SAVE_CONFIG"]["ROOT_PATH"])
        self.captured_frame_no = self._current_captured_frame_num()


    def _generate_path(self,root_path):
        """ 데이터 저장소 경로 생성"""
        PHASE = "training"
        self.OUTPUT_FOLDER = os.path.join(root_path, PHASE)
        folders = ['calib', 'image', 'kitti_label', 'carla_label', 'velodyne', 'custom', 'depth']

        for folder in folders:
            directory = os.path.join(self.OUTPUT_FOLDER, folder)
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.LIDAR_PATH = os.path.join(self.OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
        self.KITTI_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'kitti_label/{0:06}.txt')
        self.CARLA_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'carla_label/{0:06}.txt')
        self.IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'image/{0:06}.png')
        self.CALIBRATION_PATH = os.path.join(self.OUTPUT_FOLDER, 'calib/{0:06}.txt')
        self.CUSTOM_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'custom/{0:06}.txt')
        self.DEPTH_PATH = os.path.join(self.OUTPUT_FOLDER, 'depth/{0:06}.png')


    def _current_captured_frame_num(self):
        """폴더의 데이터 양 가져오기"""
        label_path = os.path.join(self.OUTPUT_FOLDER, 'kitti_label/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print("현재 {} 데이터가 있습니다".format(num_existing_data_files))
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(
                self.OUTPUT_FOLDER))
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

    def save_training_files(self, data):

        lidar_fname = self.LIDAR_PATH.format(self.captured_frame_no)
        kitti_label_fname = self.KITTI_LABEL_PATH.format(self.captured_frame_no)
        carla_label_fname = self.CARLA_LABEL_PATH.format(self.captured_frame_no)
        img_fname = self.IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = self.CALIBRATION_PATH.format(self.captured_frame_no)

        # 어디에 저장할지
        custom_label_fname = self.CUSTOM_LABEL_PATH.format(self.captured_frame_no)
        depth_fname = self.DEPTH_PATH.format(self.captured_frame_no)

        for agent, dt in data["agents_data"].items():

            camera_transform= config_to_trans(self.cfg["SENSOR_CONFIG"]["RGB"]["TRANSFORM"])
            lidar_transform = config_to_trans(self.cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"])

            save_ref_files(self.OUTPUT_FOLDER, self.captured_frame_no)
            save_image_data(img_fname, dt["sensor_data"][0])
            save_label_data(kitti_label_fname, dt["kitti_datapoints"])
            save_label_data(carla_label_fname, dt['carla_datapoints'])
            save_label_data(custom_label_fname, dt["custom_datapoints"])
            save_calibration_matrices([camera_transform, lidar_transform], calib_filename, dt["intrinsic"])
            save_lidar_data(lidar_fname, dt["sensor_data"][2])

            save_image_data(depth_fname, dt["sensor_data"][1], cc.LogarithmicDepth)
        self.captured_frame_no += 1