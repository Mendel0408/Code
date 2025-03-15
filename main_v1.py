# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
import csv
import glob
import math
import logging
from pyproj import Transformer
from scipy.spatial import distance
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay
from osgeo import gdal
import json
import geopandas as gpd
from shapely.geometry import Polygon
import os
import re
import random

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 验证字体是否存在
font_path = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, filename='debug.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

# 坐标系统统一化
class GeoCoordTransformer:
    def __init__(self):
        self.to_utm = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)
        self.to_wgs84 = Transformer.from_crs("epsg:32650", "epsg:4326", always_xy=True)

    def wgs84_to_utm(self, lon, lat):  # 注意顺序：先经度，后纬度
        try:
            easting, northing = self.to_utm.transform(lon, lat)
            if np.isinf(easting) or np.isinf(northing):
                raise ValueError('Invalid UTM coordinates')
            return easting, northing
        except Exception as e:
            raise

    def utm_to_wgs84(self, easting, northing):
        try:
            lon, lat = self.to_wgs84.transform(easting, northing)  # 注意顺序：先经度，后纬度
            if np.isinf(lat) or np.isinf(lon):
                raise ValueError('Invalid WGS84 coordinates')
            return lon, lat
        except Exception as e:
            raise

geo_transformer = GeoCoordTransformer()

# 加载DEM数据
def load_dem_data(dem_file):
    dem_dataset = gdal.Open(dem_file)
    if dem_dataset is None:
        raise RuntimeError(f"无法加载 DEM 文件: {dem_file}")

    dem_array = dem_dataset.ReadAsArray()
    gt = dem_dataset.GetGeoTransform()
    dem_x = np.arange(dem_array.shape[1]) * gt[1] + gt[0]
    dem_y = np.arange(dem_array.shape[0]) * gt[5] + gt[3]

    # 新增部分：计算 DEM 四角点在 UTM 下的坐标范围
    corners = [
        (dem_x.min(), dem_y.min()),
        (dem_x.min(), dem_y.max()),
        (dem_x.max(), dem_y.min()),
        (dem_x.max(), dem_y.max())
    ]
    utm_x_list = []
    utm_y_list = []
    for lon, lat in corners:
        try:
            easting, northing = geo_transformer.wgs84_to_utm(lon, lat)
            utm_x_list.append(easting)
            utm_y_list.append(northing)
        except Exception as e:
            logging.error(f"转换 DEM 坐标 {lon},{lat} 到 UTM 时出错: {e}")
    dem_utm_x_range = (min(utm_x_list), max(utm_x_list)) if utm_x_list else (None, None)
    dem_utm_y_range = (min(utm_y_list), max(utm_y_list)) if utm_y_list else (None, None)

    dem_interpolator = RegularGridInterpolator((dem_y, dem_x), dem_array)
    dem_data = {
        'interpolator': dem_interpolator,
        'x_range': (dem_x.min(), dem_x.max()),
        'y_range': (dem_y.min(), dem_y.max()),
        'utm_x_range': dem_utm_x_range,
        'utm_y_range': dem_utm_y_range,
        'data': dem_array
    }
    logging.debug(f"DEM 范围: 经度 {dem_data['x_range']}, 纬度 {dem_data['y_range']}")
    logging.debug(f"DEM UTM 范围: 东距 {dem_data['utm_x_range']}, 北距 {dem_data['utm_y_range']}")
    return dem_data

# 使用PnP算法进行相机姿态估计
def estimate_camera_pose(pos3d, pixels, K):
    pos3d = np.asarray(pos3d, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)


    print("Camera matrix K:\n", K)

    # 可视化3D点
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(pos3d[:, 0], pos3d[:, 1], pos3d[:, 2], c='r', marker='o')
    ax.set_title('3D Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 可视化2D点
    ax2 = fig.add_subplot(122)
    ax2.scatter(pixels[:, 0], pixels[:, 1], c='b', marker='x')
    ax2.set_title('2D Points')
    ax2.set_xlabel('Pixel X')
    ax2.set_ylabel('Pixel Y')
    ax2.invert_yaxis()  # 图像坐标系的Y轴是向下的

    plt.show()

    # 使用PnP算法估计旋转向量和平移向量，并返回内点
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        pos3d, pixels, K, dist_coeffs,
        iterationsCount=5000,
        reprojectionError=30.0,
        confidence=0.99
    )
    print("Inliers:\n", inliers)
    if not success or inliers is None or len(inliers) < 6:
        print("PnP RANSAC failed or insufficient inliers.")
        return None, None, None

    rotation_vector, translation_vector = cv2.solvePnPRefineLM(pos3d[inliers], pixels[inliers], K, dist_coeffs,
                                                               rotation_vector, translation_vector)
    print(f"Rotation Vector (R):\n{rotation_vector}")
    print(f"Translation Vector (T):\n{translation_vector}")
    return rotation_vector, translation_vector, inliers

# 获取DEM数据中的海拔值
def get_dem_elevation(dem_data, coord, coord_type='utm'):
    """
    根据坐标类型获取 DEM 高程。
    :param dem_data: DEM 数据
    :param coord: 坐标 (easting, northing) 或 (lon, lat)
    :param coord_type: 坐标类型，'utm' 或 'wgs84'
    :return: 海拔高度
    """
    if coord_type == 'utm':
        lon, lat = geo_transformer.utm_to_wgs84(coord[0], coord[1])
    elif coord_type == 'wgs84':
        lon, lat = coord
    else:
        raise ValueError("Invalid coord_type. Must be 'utm' or 'wgs84'.")

    # 插值器构造时使用的坐标顺序为 (lat, lon)
    dem_elev = dem_data['interpolator']((lat, lon))
    return dem_elev

# 将像素坐标转换为射线
def pixel_to_ray(pixel_x, pixel_y, K, R, ray_origin):
    """
    将像素坐标 (pixel_x, pixel_y) 转换为射线方向.

    参数:
      pixel_x, pixel_y -- 像素坐标
      K -- 根据物理参数换算到像素单位的内参矩阵
      R -- 旋转矩阵，要求转换后可将相机坐标转换至UTM坐标系
      ray_origin -- 相机在UTM坐标系下的位置

    返回:
      (ray_origin, ray_direction) 其中 ray_direction 为UTM坐标系下的单位向量
    """
    # 构建齐次像素向量
    pixel_homogeneous = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
    print("【DEBUG】像素齐次向量:", pixel_homogeneous)

    # 计算相机坐标下的射线方向（未归一化）
    camera_ray = np.linalg.inv(K) @ pixel_homogeneous
    camera_ray /= np.linalg.norm(camera_ray)
    print("【DEBUG】相机坐标下的射线方向:", camera_ray)

    # 将相机坐标下的射线方向转换到UTM坐标系
    # 如果你的流程中 R 已为从物体到相机坐标的转换，应使用 R.T 进行转换
    utm_ray = R.T @ camera_ray
    utm_ray /= np.linalg.norm(utm_ray)

    return ray_origin, utm_ray


def calculate_weights(input_pixel, control_points, max_weight=1, knn_weight=30):
    weights = []
    input_pixel = np.array(input_pixel, dtype=np.float64)  # 确保 input_pixel 是浮点数类型
    distances = []
    for cp in control_points:
        pixel = np.array(cp['pixel'], dtype=np.float64)  # 确保 pixel 是浮点数类型
        distance = np.linalg.norm(input_pixel - pixel)
        distances.append(distance)
        weight = min(1.0 / distance if distance != 0 else 1.0, max_weight)  # 限制权重最大值
        weights.append(weight)

    # 找到距离最近的控制点并提升其权重
    min_distance_index = np.argmin(distances)
    weights[min_distance_index] *= knn_weight

    # 输出每个控制点的距离和权重信息
    for i, cp in enumerate(control_points):
        logging.debug(f"【DEBUG】控制点 {cp['symbol']} 的距离: {distances[i]}, 权重: {weights[i]}")

    return np.array(weights)


def compute_optimization_factors(control_points, K, R, ray_origin):
    optimization_factors = []
    for cp in control_points:
        true_geo = np.array(cp['pos3d'], dtype=np.float64)
        ideal_direction = true_geo - ray_origin
        print(f"【DEBUG】归一化前的理想UTM射线方向: {ideal_direction}")
        norm_ideal = np.linalg.norm(ideal_direction)
        if norm_ideal == 0:
            continue
        ideal_direction /= norm_ideal
        _, computed_ray = pixel_to_ray(cp['pixel'][0], cp['pixel'][1], K, R, ray_origin)
        computed_ray /= np.linalg.norm(computed_ray)
        optimization_factor_x = ideal_direction[0] / computed_ray[0]
        optimization_factor_y = ideal_direction[1] / computed_ray[1]
        optimization_factor_z = ideal_direction[2] / computed_ray[2]

        # 增加异常值检测和过滤
        if abs(optimization_factor_x) > 2 or abs(optimization_factor_y) > 2 or abs(optimization_factor_z) > 2:
            print(f"【警告】控制点 {cp['symbol']} 的优化因子异常，已过滤: ({optimization_factor_x}, {optimization_factor_y}, {optimization_factor_z})")
            continue

        optimization_factors.append((optimization_factor_x, optimization_factor_y, optimization_factor_z))
        cp['factors'] = (optimization_factor_x, optimization_factor_y, optimization_factor_z)  # 保存优化因子到控制点
        print(f"【DEBUG】控制点 {cp['symbol']} 的理想UTM射线方向: {ideal_direction}")
        print(f"【DEBUG】像素射线方向: {computed_ray}")
        print(f"【DEBUG】控制点 {cp['symbol']} 的优化因子: ({optimization_factor_x}, {optimization_factor_y}, {optimization_factor_z})")
    return optimization_factors

def weighted_average_optimization_factors(factors, weights):
    # 将权重归一化
    normalized_weights = weights / np.sum(weights)
    print(f"【DEBUG】归一化权重: {normalized_weights}")
    weighted_factors = np.average(factors, axis=0, weights=normalized_weights)
    return weighted_factors

# 计算射线与DEM的交点
def ray_intersect_dem(ray_origin, ray_direction, dem_data, max_search_dist=5000, step=1):
    current_pos = np.array(ray_origin, dtype=np.float64)
    step_count = 0  # 初始化步进计数器
    for _ in range(int(max_search_dist / step)):
        logging.debug(f"【DEBUG】当前UTM坐标: {current_pos}, 当前射线方向: {ray_direction}")
        current_easting = current_pos[0]
        current_northing = current_pos[1]
        lon, lat = geo_transformer.utm_to_wgs84(current_easting, current_northing)
        try:
            dem_elev = dem_data['interpolator']((lat, lon))
        except Exception as e:
            logging.error(f"【错误】插值时出错: {e}")
            return None, step_count
        logging.debug(f"【DEBUG】DEM海拔: {dem_elev}, 当前高度: {current_pos[2]}")

        if step_count >= 50 and current_pos[2] <= dem_elev + 0.5 :
            return np.array([current_easting, current_northing, current_pos[2]]), step_count

        current_pos[0] += step * ray_direction[0]
        current_pos[1] += step * ray_direction[1]
        current_pos[2] += step * ray_direction[2]
        step_count += 1  # 增加步进计数器

    return None, step_count

# 输入像素坐标，输出地理坐标
def pixel_to_geo(pixel_coord, K, R, ray_origin, dem_data, control_points, optimization_factors):
    # 计算权重
    weights = calculate_weights(pixel_coord, control_points)
    # 计算加权优化因子
    weighted_optimization_factors = weighted_average_optimization_factors(optimization_factors, weights)
    print(f"【DEBUG】加权优化因子: {weighted_optimization_factors}")
    # 计算射线方向
    ray_origin, ray_direction = pixel_to_ray(pixel_coord[0], pixel_coord[1], K, R, ray_origin)
    print(f"【DEBUG】初始射线方向: {ray_direction}")
    # 应用优化因子校正射线方向的Z分量
    optimized_ray_direction = np.array([
        ray_direction[0] * weighted_optimization_factors[0],
        ray_direction[1] * weighted_optimization_factors[1],
        ray_direction[2] * weighted_optimization_factors[2]
    ])
    print(f"【DEBUG】优化射线方向: {optimized_ray_direction}")
    # 归一化校正后的射线方向
    final_ray_direction = optimized_ray_direction / np.linalg.norm(optimized_ray_direction)
    print(f"【DEBUG】最终射线方向: {final_ray_direction}")
    # 计算射线与DEM的交点
    geo_coord, total_steps = ray_intersect_dem(ray_origin, final_ray_direction, dem_data)
    print(f"【DEBUG】地理坐标: {geo_coord}")
    print(f"【DEBUG】射线步进总步数: {total_steps}")

    return geo_coord, total_steps

# **********
# read data from the features file
# **********
def read_points_data(filename, pixel_x, pixel_y, scale):
    geo_transformer = GeoCoordTransformer()
    with open(filename, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
        pixels = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                names = row
                indx = names.index(pixel_x)
                indy = names.index(pixel_y)
            else:
                line_count += 1
                symbol = row[1]
                name = row[2]
                pixel = np.array([int(row[indx]), int(row[indy])]) / scale
                longitude = float(row[4])
                latitude = float(row[5])
                elevation = float(row[6])
                # 跳过当前处理照片中像素坐标为0,0的点
                if int(row[indx]) == 0 and int(row[indy]) == 0:
                    continue
                pixels.append(pixel)

                easting, northing = geo_transformer.wgs84_to_utm(longitude, latitude)
                pos3d = np.array([easting, northing, elevation])
                print(f"Processed Point - Symbol: {symbol}, Name: {name}, Pixel: {pixel}, Lon: {longitude}, Lat: {latitude}, Easting: {easting}, Northing: {northing}, Elevation: {elevation}")
                rec = {'symbol': symbol, 'pixel': pixel, 'pos3d': pos3d}
                recs.append(rec)
        return recs

# 将边界像素坐标转换为地理坐标
def convert_boundary_to_geo(json_data, K, R, ray_origin, dem_data, control_points, optimization_factors):
    boundary_points = {}
    boundary_geo_coords = {}

    for obj in json_data['objects']:
        group = obj['group']
        category = re.sub(r'[^a-zA-Z0-9]', '', obj['category'])
        key = (group, category)

        if key not in boundary_geo_coords:
            boundary_geo_coords[key] = []
            boundary_points[key] = []

        boundary_points_obj = obj['segmentation']
        for pixel_x, pixel_y in boundary_points_obj:
            geo_coord = pixel_to_geo([pixel_x, pixel_y], K, R, ray_origin, dem_data, control_points, optimization_factors)
            if geo_coord is not None:
                boundary_geo_coords[key].append(geo_coord)
                boundary_points[key].append((pixel_x, pixel_y))

    return boundary_geo_coords, boundary_points

# 生成csv
def save_boundary_to_csv(boundary_geo_coords, boundary_points, csv_file='boundary_points_geo.csv'):
    csv_data = []

    for (group, category), coords in boundary_geo_coords.items():
        for i, coord in enumerate(coords):
            if len(coord) < 3:
                logging.error(f"Invalid coordinate {coord} for category {category} and group {group}")
                continue
            pixel_x, pixel_y = boundary_points[(group, category)][i]
            csv_data.append([category, group, pixel_x, pixel_y, coord[0], coord[1], coord[2]])

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'group', 'pixel_x', 'pixel_y', 'geo_x', 'geo_y', 'geo_z'])
        writer.writerows(csv_data)

    print(f"CSV文件已保存到 {csv_file}")

# 生成shp
def save_boundary_to_shapefiles(boundary_geo_coords, json_data, output_dir):
    name = json_data['info']['name']

    for (group, category), group_coords in boundary_geo_coords.items():
        if len(group_coords) < 3:
            print(f"Group {group}, Category {category} 边界点数量少于3个，生成Polygon失败")
            continue

        group_coords = [(coord[0], coord[1]) for coord in group_coords if len(coord) >= 2]

        attributes = []
        geometry = []

        polygon = Polygon(group_coords)
        geometry.append(polygon)
        attributes.append({
            'group': group,
            'name': name,
            'category': category,
            'area': polygon.area,
            'perimeter': polygon.length,
        })

        gdf = gpd.GeoDataFrame(attributes, geometry=geometry)
        gdf.set_crs(epsg=32650, inplace=True)

        sanitized_category = re.sub(r'[^a-zA-Z0-9]', '', category)
        output_shp_file = os.path.join(output_dir, f"{sanitized_category}_{group}_boundary.shp")
        gdf.to_file(output_shp_file, driver='ESRI Shapefile')
        print(f"Shapefile已保存到 {output_shp_file}")

# **********
# Main function
# **********
def do_it(image_name, json_file, features, camera_locations, pixel_x, pixel_y, output, scale, dem_file):
    # 确保工作目录为脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 检查图像文件是否存在
    image_path = os.path.join("historical photos", image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    im = cv2.imread(image_path)
    im2 = np.copy(im)
    im[:, :, 0] = im2[:, :, 2]
    im[:, :, 1] = im2[:, :, 1]
    im[:, :, 2] = im2[:, :, 0]

    plt.figure(figsize=(11.69, 8.27))
    plt.imshow(im)

    # 加载DEM数据（DEM范围为UTM）
    dem_data = load_dem_data(dem_file)

    recs = read_points_data(features, pixel_x, pixel_y, scale)

    # 设置 K 矩阵
    width, height = im.shape[1], im.shape[0]
    cx = width / 2
    cy = height / 2
    # 相机物理参数（单位：mm）
    focal_length_mm = 180.0
    sensor_width_mm = 102.0
    sensor_height_mm = 127.0
    # 根据物理参数将焦距换算为像素单位：
    fx = focal_length_mm / sensor_width_mm * width
    fy = focal_length_mm / sensor_height_mm * height

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)

    # 使用PnP算法估计 R 和 T，得到的相机位置camera_origin为UTM数据
    pos3d = np.array([rec['pos3d'] for rec in recs])
    pixels = np.array([rec['pixel'] for rec in recs])
    R, t, inliers = estimate_camera_pose(pos3d, pixels, K)
    if R is None or t is None:
        print("Failed to estimate camera pose using PnP.")
        return
    R, _ = cv2.Rodrigues(R)
    print(f"【DEBUG】转换后的旋转矩阵 R: \n{R}")

    # 使用所有 pos3d 中的点作为控制点
    control_points = [
        {
            'pixel': recs[idx]['pixel'],
            'pos3d': recs[idx]['pos3d'],
            'dem_data': dem_data,
            'symbol': recs[idx]['symbol']
        }
        for idx in range(len(recs))
    ]

    # 使用PnP求解后的相机位置（camera_origin为UTM坐标）
    camera_origin = -R.T @ t.flatten()
    print(f"【DEBUG】PnP求解的相机位置 (UTM): {camera_origin}")

    # 修正相机位置海拔
    dem_elev = get_dem_elevation(dem_data,  camera_origin, coord_type='utm')
    corrected_camera_origin = np.array([camera_origin[0], camera_origin[1], dem_elev + 1.5 ])

    # 直接使用修正后的camera_origin作为ray_origin
    ray_origin = corrected_camera_origin
    print(f"【DEBUG】用于射线方向计算的相机位置: {ray_origin}")

    # 校验相机位置是否在 DEM 的 UTM 覆盖范围内，直接使用 dem_data 中的 utm_x_range 和 utm_y_range
    tol = 1e-5
    utm_x_range = dem_data['utm_x_range']
    utm_y_range = dem_data['utm_y_range']
    if not (utm_x_range[0] - tol <= camera_origin[0] <= utm_x_range[1] + tol and
            utm_y_range[0] - tol <= camera_origin[1] <= utm_y_range[1] + tol):
        print(f"【错误】ray_origin {camera_origin} 超出 DEM 范围")
        print(f"【DEBUG】DEM 范围 (UTM): 经度 {utm_x_range}, 纬度 {utm_y_range}")
        return

    # 计算每个控制点的优化因子
    optimization_factors = compute_optimization_factors(control_points, K, R, ray_origin)

    while True:
        try:
            input_pixel_str = input("请输入像素坐标 (x, y) 或输入 'exit' 退出: ").strip()
            if input_pixel_str.lower() == 'exit':
                break

            pixel_values = input_pixel_str.replace(" ", "").replace("，", ",").split(",")
            if len(pixel_values) != 2:
                print("输入格式错误，例：755,975")
                continue

            input_pixel_x, input_pixel_y = map(float, pixel_values)
            input_pixel = [input_pixel_x, input_pixel_y]  # 使用输入的像素坐标

            geo_coord, total_steps = pixel_to_geo(input_pixel, K, R, ray_origin, dem_data, control_points, optimization_factors)
            if geo_coord is not None:
                print(f"像素坐标 ({input_pixel_x}, {input_pixel_y}) 对应的UTM地理坐标:")
                print(f"Easting: {geo_coord[0]:.2f}, Northing: {geo_coord[1]:.2f}, 高度: {geo_coord[2]:.2f}")
                print(f"射线步进总步数: {total_steps}")
            else:
                print(f"无法找到 ({input_pixel_x}, {input_pixel_y}) 对应的地理坐标，请检查输入或 DEM 数据。")

        except ValueError as e:
            print(f"输入格式错误: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")

    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 将边界点转换为地理坐标
    boundary_geo_coords, boundary_points = convert_boundary_to_geo(json_data, K, R, ray_origin, dem_data, control_points, optimization_factors)

    # 保存为csv
    save_boundary_to_csv(boundary_geo_coords, boundary_points)

    # 保存为Shapefile
    save_boundary_to_shapefiles(boundary_geo_coords, json_data, "output_shapefiles")

# 主函数处理多个图像
def main():
    images_info = [
        {
            "image_name": "1898.jpg",
            "json_file": "1898.json",
            "features": "feature_points_with_annotations.csv",
            "camera_locations": "potential_camera_locations.csv",
            "pixel_x": "Pixel_x_1898.jpg",
            "pixel_y": "Pixel_y_1898.jpg",
            "output": "zOutput_1898.png",
            "scale": 1.0,
            "dem_file": "dem_dx.tif"
        },
        {
            "image_name": "1900-1910.jpg",
            "json_file": "1900-1910.json",
            "features": "feature_points_with_annotations.csv",
            "camera_locations": "potential_camera_locations.csv",
            "pixel_x": "Pixel_x_1900-1910.jpg",
            "pixel_y": "Pixel_y_1900-1910.jpg",
            "output": "zOutput_1900-1910.png",
            "scale": 1.0,
            "dem_file": "dem_dx.tif"
        },
        # 添加更多图像的信息
        # {
        #     "image_name": "another_image.jpg",
        #     "json_file": "another.json",
        #     "features": "another_features.csv",
        #     "camera_locations": "another_camera_locations.csv",
        #     "pixel_x": "Pixel_x_another.jpg",
        #     "pixel_y": "Pixel_y_another.jpg",
        #     "output": "zOutput_another.png",
        #     "scale": 1.0,
        #     "dem_file": "dem_data_another.tif"
        # }
    ]

    # 指定要处理的图像信息
    target_info = images_info[0]  # 修改此索引以选择要处理的图像，例如 0 表示处理第一个图像

    do_it(
        target_info["image_name"],
        target_info["json_file"],
        target_info["features"],
        target_info["camera_locations"],
        target_info["pixel_x"],
        target_info["pixel_y"],
        target_info["output"],
        target_info["scale"],
        target_info["dem_file"]
    )

if __name__ == "__main__":
    main()

print('**********************')
# print ('ret: ')
# print (ret)
# print ('mtx: ')
# print (mtx)
# print ('dist: ')
# print (dist)
# print('rvecs: ')
# print(rvecs)
# print ('tvecs: ')
# print(tvecs)

print('Done!')
