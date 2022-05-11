import sys

import numpy as np
from numpy.linalg import inv

from config import cfg_from_yaml_file
from data_descriptor import KittiDescriptor, CarlaDescriptor, CustomDescriptor
from image_converter import depth_to_array, to_rgb_array
import math
from visual_utils import draw_3d_bounding_box

import glob
import os

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

cfg = cfg_from_yaml_file("configs.yaml")

MAX_RENDER_DEPTH_IN_METERS = cfg["FILTER_CONFIG"]["MAX_RENDER_DEPTH_IN_METERS"]
MIN_VISIBLE_VERTICES_FOR_RENDER = cfg["FILTER_CONFIG"]["MIN_VISIBLE_VERTICES_FOR_RENDER"]
MAX_OUT_VERTICES_FOR_RENDER = cfg["FILTER_CONFIG"]["MAX_OUT_VERTICES_FOR_RENDER"]
WINDOW_WIDTH = cfg["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_x"]
WINDOW_HEIGHT = cfg["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_y"]


def objects_filter(data):
    # environment_objects = data["environment_objects"]
    agents_data = data["agents_data"]
    actors = data["actors"]
    actors = [x for x in actors if x.type_id.find("vehicle") != -1 or x.type_id.find("walker") != -1]
    for agent, dataDict in agents_data.items():
        intrinsic = dataDict["intrinsic"]
        extrinsic = dataDict["extrinsic"]
        sensors_data = dataDict["sensor_data"]
        kitti_datapoints = []
        carla_datapoints = []
        custom_datapoints = []
        rgb_image = to_rgb_array(sensors_data[0])
        image = rgb_image.copy()
        depth_data = sensors_data[1]

        # data["agents_data"][agent]["visible_environment_objects"] = []
        # for obj in environment_objects:
        #     custom_datapoint, kitti_datapoint, carla_datapoint = is_visible_by_bbox(agent, obj, image, depth_data, intrinsic, extrinsic, 0)
        #     if kitti_datapoint is not None:
        #         data["agents_data"][agent]["visible_environment_objects"].append(obj)
        #         kitti_datapoints.append(kitti_datapoint)
        #         carla_datapoints.append(carla_datapoint)
        #         custom_datapoints.append(custom_datapoint)

        data["agents_data"][agent]["visible_actors"] = []

        for act in actors:
            custom_datapoint, kitti_datapoint, carla_datapoint = is_visible_by_bbox(agent, act, image, depth_data, intrinsic, extrinsic, 1)
            if kitti_datapoint is not None:
                data["agents_data"][agent]["visible_actors"].append(act)
                kitti_datapoints.append(kitti_datapoint)
                carla_datapoints.append(carla_datapoint)
                custom_datapoints.append(custom_datapoint)


        data["agents_data"][agent]["rgb_image"] = image
        data["agents_data"][agent]["kitti_datapoints"] = kitti_datapoints
        data["agents_data"][agent]["carla_datapoints"] = carla_datapoints
        data["agents_data"][agent]["custom_datapoints"] = custom_datapoints
    return data


def is_visible_by_bbox(agent, obj, rgb_image, depth_data, intrinsic, extrinsic, tags=0):
    obj_transform = obj.transform if isinstance(obj, carla.EnvironmentObject) else obj.get_transform()
    obj_bbox = obj.bounding_box
    if isinstance(obj, carla.EnvironmentObject):
        vertices_pos2d = bbox_2d_from_agent(intrinsic, extrinsic, obj_bbox, obj_transform, 0)
    else:
        vertices_pos2d = bbox_2d_from_agent(intrinsic, extrinsic, obj_bbox, obj_transform, 1)
    depth_image = depth_to_array(depth_data)
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(vertices_pos2d, depth_image)
    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < MAX_OUT_VERTICES_FOR_RENDER:
        obj_tp = obj_type(obj)
        midpoint = midpoint_from_agent_location(obj_transform.location, extrinsic)
        bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
        rotation_y = get_relative_rotation_y(agent.get_transform().rotation, obj_transform.rotation) % math.pi
        ext = obj.bounding_box.extent
        truncated = num_vertices_outside_camera / 8
        if num_visible_vertices >= 6:
            occluded = 0
        elif num_visible_vertices >= 4:
            occluded = 1
        else:
            occluded = 2

        velocity = "0 0 0" if isinstance(obj, carla.EnvironmentObject) else\
            "{} {} {}".format(obj.get_velocity().x, obj.get_velocity().y, obj.get_velocity().z)
        acceleration = "0 0 0" if isinstance(obj, carla.EnvironmentObject) else \
            "{} {} {}".format(obj.get_acceleration().x, obj.get_acceleration().y, obj.get_acceleration().z)
        angular_velocity = "0 0 0" if isinstance(obj, carla.EnvironmentObject) else\
            "{} {} {}".format(obj.get_angular_velocity().x, obj.get_angular_velocity().y, obj.get_angular_velocity().z)
        # draw_3d_bounding_box(rgb_image, vertices_pos2d)


        # blueprint.id
        if tags==0: # obj
            obj_id = -1
        else:
            obj_id = obj.type_id.split(".")
            if obj_id[0] == "walker":
                obj_id = int(obj_id[2])
            else:
                obj_id = -1
            # obj_id = int(obj.type_id.split(".")[2])

        custom_data = CustomDescriptor()
        custom_data.set_bp_id(obj_id)
        custom_data.set_truncated(truncated)
        custom_data.set_occlusion(occluded)
        custom_data.set_bbox(bbox_2d)
        custom_data.set_3d_object_dimensions(ext)
        custom_data.set_type(obj_tp)
        custom_data.set_3d_object_location(midpoint)
        custom_data.set_rotation_y(rotation_y)

        kitti_data = KittiDescriptor()
        kitti_data.set_truncated(truncated)
        kitti_data.set_occlusion(occluded)
        kitti_data.set_bbox(bbox_2d)
        kitti_data.set_3d_object_dimensions(ext)
        kitti_data.set_type(obj_tp)
        kitti_data.set_3d_object_location(midpoint)
        kitti_data.set_rotation_y(rotation_y)

        carla_data = CarlaDescriptor()
        carla_data.set_type(obj_tp)
        carla_data.set_velocity(velocity)
        carla_data.set_acceleration(acceleration)
        carla_data.set_angular_velocity(angular_velocity)
        return custom_data, kitti_data, carla_data
    return None, None, None

def obj_type(obj):
    if isinstance(obj, carla.EnvironmentObject):
        return obj.type
    else:
        if obj.type_id.find('walker') is not -1:
            return 'Pedestrian'
        if obj.type_id.find('vehicle') is not -1:
            return 'Car'
        return None

def get_relative_rotation_y(agent_rotation, obj_rotation):
    """ 返回actor和camera在rotation yaw的相对角度 """

    rot_agent = agent_rotation.yaw
    rot_car = obj_rotation.yaw
    return degrees_to_radians(rot_agent - rot_car)


def bbox_2d_from_agent(intrinsic_mat, extrinsic_mat, obj_bbox, obj_transform, obj_tp):
    bbox = vertices_from_extension(obj_bbox.extent)
    if obj_tp == 1:
        bbox_transform = carla.Transform(obj_bbox.location, obj_bbox.rotation)
        bbox = transform_points(bbox_transform, bbox)
    else:
        box_location = carla.Location(obj_bbox.location.x-obj_transform.location.x,
                                      obj_bbox.location.y-obj_transform.location.y,
                                      obj_bbox.location.z-obj_transform.location.z)
        box_rotation = obj_bbox.rotation
        bbox_transform = carla.Transform(box_location, box_rotation)
        bbox = transform_points(bbox_transform, bbox)
    # 세계 좌표계에서 bbox의 점 좌표를 얻습니다.
    bbox = transform_points(obj_transform, bbox)
    # 세계 좌표계에서 bbox의 8개 점을 2차원 이미지로 변환
    vertices_pos2d = vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat)
    return vertices_pos2d


def vertices_from_extension(ext):
    """ 以自身为原点的八个点的坐标 """
    return np.array([
        [ext.x, ext.y, ext.z],  # Top left front
        [- ext.x, ext.y, ext.z],  # Top left back
        [ext.x, - ext.y, ext.z],  # Top right front
        [- ext.x, - ext.y, ext.z],  # Top right back
        [ext.x, ext.y, - ext.z],  # Bottom left front
        [- ext.x, ext.y, - ext.z],  # Bottom left back
        [ext.x, - ext.y, - ext.z],  # Bottom right front
        [- ext.x, - ext.y, - ext.z]  # Bottom right back
    ])


def transform_points(transform, points):
    """ 자신을 원점으로 하는 8점의 좌표 """
    # 转置
    points = points.transpose()
    # [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]  (4,8)
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # transform.get_matrix() 获取当前坐标系向相对坐标系的旋转矩阵
    points = np.mat(transform.get_matrix()) * points
    # 返回前三行
    return points[0:3].transpose()


def vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat):
    """세계 좌표계에서 bbox의 점을 카메라에 투영하여 2차원 그림의 좌표와 점의 깊이를 얻습니다."""
    vertices_pos2d = []
    for vertex in bbox:
        # 세계 좌표계에서 점의 벡터를 얻습니다.
        pos_vector = vertex_to_world_vector(vertex)
        # 점의 세계 좌표를 카메라 좌표계로 변환
        transformed_3d_pos = proj_to_camera(pos_vector, extrinsic_mat)
        # 점의 카메라 좌표를 2D 이미지의 좌표로 변환
        pos2d = proj_to_2d(transformed_3d_pos, intrinsic_mat)
        # 포인트 실제 깊이
        vertex_depth = pos2d[2]
        # 이미지에서 점의 좌표
        x_2d, y_2d = pos2d[0], pos2d[1]
        vertices_pos2d.append((y_2d, x_2d, vertex_depth))
    return vertices_pos2d


def vertex_to_world_vector(vertex):
    """ 칼라 월드 벡터의 정점 좌표를 반환합니다. (X, Y, Z, 1) (4,1)"""
    return np.array([
        [vertex[0, 0]],  # [[X,
        [vertex[0, 1]],  # Y,
        [vertex[0, 2]],  # Z,
        [1.0]  # 1.0]]
    ])


def calculate_occlusion_stats(vertices_pos2d, depth_image):
    """ 기능: bbox 의 8개 꼭짓점에서 실제 보이는 점을 필터링합니다."""
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    for y_2d, x_2d, vertex_depth in vertices_pos2d:
        # 포인트는 가시 범위에 있으며 이미지 범위를 초과하지 않습니다.
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas((y_2d, x_2d)):
            is_occluded = point_is_occluded(
                (y_2d, x_2d), vertex_depth, depth_image)
            if not is_occluded:
                num_visible_vertices += 1
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def point_in_canvas(pos):
    if (pos[0] >= 0) and (pos[0] < WINDOW_HEIGHT) and (pos[1] >= 0) and (pos[1] < WINDOW_WIDTH):
        return True
    return False


def point_is_occluded(point, vertex_depth, depth_image):
    y, x = map(int, point)
    from itertools import product
    neigbours = product((1, -1), repeat=2)
    is_occluded = []
    for dy, dx in neigbours:
        if point_in_canvas((dy + y, dx + x)):
            # 점에서 영상까지의 거리가 깊이 영상에 해당하는 깊이의 깊이 값보다 큰지 판단
            if depth_image[y + dy, x + dx] < vertex_depth:
                is_occluded.append(True)
            else:
                is_occluded.append(False)
    # 4개의 인접 포인트가 모두 깊이 이미지 값보다 크면 포인트가 가려집니다. true를 반환
    return all(is_occluded)


def midpoint_from_agent_location(location, extrinsic_mat):
    """ 월드 좌표계에서 에이전트의 중심점을 카메라 좌표계로 변환 """
    midpoint_vector = np.array([
        [location.x],  # [[X,
        [location.y],  # Y,
        [location.z],  # Z,
        [1.0]  # 1.0]]
    ])
    transformed_3d_midpoint = proj_to_camera(midpoint_vector, extrinsic_mat)
    return transformed_3d_midpoint


def camera_intrinsic(width, height):
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    return k


def proj_to_camera(pos_vector, extrinsic_mat):
    """ 기능: 점의 세계 좌표를 카메라 좌표계 로 변환합니다."""
    # inv求逆矩阵
    transformed_3d_pos = np.dot(inv(extrinsic_mat), pos_vector)
    return transformed_3d_pos


def proj_to_2d(camera_pos_vector, intrinsic_mat):
    """카메라 좌표계에서 점의 3D 좌표를 이미지에 투영"""
    cords_x_y_z = camera_pos_vector[:3, :]
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    pos2d = np.dot(intrinsic_mat, cords_y_minus_z_x)
    # normalize the 2D points
    pos2d = np.array([
        pos2d[0] / pos2d[2],
        pos2d[1] / pos2d[2],
        pos2d[2]
    ])
    return pos2d


def filter_by_distance(data_dict, dis):
    actors = data_dict["actors"]
    for agent,_ in data_dict["agents_data"].items():
        data_dict["actors"] = [act for act in actors if
                                            distance_between_locations(act.get_location(), agent.get_location())<dis]


def distance_between_locations(location1, location2):
    return math.sqrt(pow(location1.x-location2.x, 2)+pow(location1.y-location2.y, 2))

def calc_projected_2d_bbox(vertices_pos2d):
    """ 8개의 꼭짓점의 이미지 좌표를 기반으로 2차원 bbox의 왼쪽 위 좌표와 오른쪽 아래 좌표를 계산합니다. """
    legal_pos2d = list(filter(lambda x: x is not None, vertices_pos2d))
    y_coords, x_coords = [int(x[0][0]) for x in legal_pos2d], [
        int(x[1][0]) for x in legal_pos2d]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coorsds), max(y_coords)
    return [min_x, min_y, max_x, max_y]

def degrees_to_radians(degrees):
    return degrees * math.pi / 180