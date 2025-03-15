import csv
import cv2
import os
import numpy as np
from pyproj import Transformer

class GeoCoordTransformer:
    def __init__(self):
        self.to_utm = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)
        self.to_wgs84 = Transformer.from_crs("epsg:32650", "epsg:4326", always_xy=True)

    def wgs84_to_utm(self, lon, lat):
        easting, northing = self.to_utm.transform(lon, lat)
        return easting, northing

    def utm_to_wgs84(self, easting, northing):
        lon, lat = self.to_wgs84.transform(easting, northing)
        return lon, lat

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

def compute_reprojection_error(pos3d, pixels, K, dist_coeffs, rvec, tvec):
    projected_points, _ = cv2.projectPoints(pos3d, rvec, tvec, K, dist_coeffs)
    projected_points = projected_points.squeeze()
    errors = np.linalg.norm(pixels - projected_points, axis=1)
    return errors

def estimate_camera_orientation(points_data, focal_lengths, sensor_sizes, image_size):
    pos3d = np.array([rec['pos3d'] for rec in points_data], dtype=np.float64).reshape(-1, 3)
    pixels = np.array([rec['pixel'] for rec in points_data], dtype=np.float64).reshape(-1, 2)
    symbols = [rec['symbol'] for rec in points_data]
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    all_results = []

    transformer = GeoCoordTransformer()

    for focal_length in focal_lengths:
        for (sensor_width, sensor_height) in sensor_sizes:
            pixel_size_width = sensor_width / image_size[0]
            pixel_size_height = sensor_height / image_size[1]

            fx = focal_length / pixel_size_width
            fy = focal_length / pixel_size_height

            K = np.array([
                [fx, 0, image_size[0] / 2],
                [0, fy, image_size[1] / 2],
                [0, 0, 1]
            ], dtype=np.float32)

            success, initial_rotation_vector, initial_translation_vector, inliers = cv2.solvePnPRansac(
                pos3d, pixels, K, dist_coeffs, useExtrinsicGuess=False,
                iterationsCount=5000, reprojectionError=30.0, confidence=0.99
            )

            if not success or inliers is None or len(inliers) < 6:
                continue

            optimized_rotation_vector, optimized_translation_vector = cv2.solvePnPRefineLM(
                pos3d[inliers.flatten()], pixels[inliers.flatten()], K, dist_coeffs,
                initial_rotation_vector, initial_translation_vector
            )

            errors_initial = compute_reprojection_error(pos3d[inliers.flatten()], pixels[inliers.flatten()], K, dist_coeffs,
                                                        initial_rotation_vector, initial_translation_vector)
            mean_error_initial = np.mean(errors_initial)

            errors_optimized = compute_reprojection_error(pos3d[inliers.flatten()], pixels[inliers.flatten()], K, dist_coeffs,
                                                          optimized_rotation_vector, optimized_translation_vector)
            mean_error_optimized = np.mean(errors_optimized)

            R_matrix, _ = cv2.Rodrigues(optimized_rotation_vector)
            camera_origin = -R_matrix.T @ optimized_translation_vector.flatten()
            distance_to_known_origin = np.linalg.norm(camera_origin[:2] - known_camera_origin[:2])

            inlier_symbols = [symbols[idx] for idx in inliers.flatten()]

            all_results.append((distance_to_known_origin, inlier_symbols, mean_error_initial, mean_error_optimized, K, focal_length, sensor_width, sensor_height, camera_origin))

    all_results.sort(key=lambda x: x[3])

    filtered_results = [result for result in all_results if result[0] <= 150]

    if len(filtered_results) < 5:
        print("Filtered results are less than 5. Showing available results:")
    else:
        print("\nTop 5 camera origins sorted by optimized reprojection error (with WGS84 coordinates):\n")

    for i in range(min(5, len(filtered_results))):
        distance, inliers, mean_error_initial, mean_error_optimized, K, focal_length, sensor_width, sensor_height, camera_origin = filtered_results[i]
        lon, lat = transformer.utm_to_wgs84(camera_origin[0], camera_origin[1])
        print(f"Rank {i+1}:")
        print(f"Distance to known camera origin: {distance:.2f} meters")
        print(f"Inliers: {inliers}")
        print(f"Initial reprojection error: {mean_error_initial:.2f} pixels")
        print(f"Optimized reprojection error: {mean_error_optimized:.2f} pixels")
        print(f"Focal Length: {focal_length} mm, Sensor Size: {sensor_width}x{sensor_height} mm")
        print(f"K Matrix:\n{K}")
        print(f"Camera Origin (UTM): {camera_origin}")
        print(f"Camera Origin (WGS84): ({lon}, {lat}, {camera_origin[2]})\n")

    return filtered_results


def optimize_camera_center(selected_result, points_data):
    distance, inliers, mean_error_initial, mean_error_optimized, K, focal_length, sensor_width, sensor_height, camera_origin = selected_result

    # 初始相机矩阵，假设光心在图像中央
    w, h = image_size
    K[0, 2] = w / 2
    K[1, 2] = h / 2
    print("w", [w])
    print("h", [h])

    # 使用所有点进行相机标定
    pos3d_all = np.array([rec['pos3d'] for rec in points_data], dtype=np.float32)
    pixels_all = np.array([rec['pixel'] for rec in points_data], dtype=np.float32)

    print("pos3d_all", [pos3d_all])
    print("pixels_all", [pixels_all])

    # 初始化畸变系数为0
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    print(f"初始相机内参:\n{K}")

    # 使用 cv2.calibrateCamera 进行相机标定优化光心，固定焦距和畸变系数
    ret, K_optimized, dist_coeffs_optimized, rvecs, tvecs = cv2.calibrateCamera(
        [pos3d_all], [pixels_all], (w, h), K.astype(np.float32), dist_coeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_ASPECT_RATIO |
              cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    )

    print(f"Optimized Camera Matrix:\n{K_optimized}")
    print(f"Optimized Distortion Coefficients:\n{dist_coeffs_optimized}")

# 图片文件选择
img = '1898'
if img == '1898':
    image_name = '1898.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_1898.jpg'
    pixel_y = 'Pixel_y_1898.jpg'
    scale = 1.0
elif img == '1900-1910':
    image_name = '1900-1910.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_1900-1910.jpg'
    pixel_y = 'Pixel_y_1900-1910.jpg'
    scale = 1.0
elif img == '1910':
    image_name = '1910.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_1910.jpg'
    pixel_y = 'Pixel_y_1910.jpg'
    scale = 1.0
elif img == '1912s':
    image_name = '1912s.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_1912s.jpg'
    pixel_y = 'Pixel_y_1912s.jpg'
    scale = 1.0
elif img == '1915 (2)':
    image_name = '1915 (2).jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_1915 (2).jpg'
    pixel_y = 'Pixel_y_1915 (2).jpg'
    scale = 1.0
elif img == '1915':
    image_name = '1915.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_1915.jpg'
    pixel_y = 'Pixel_y_1915.jpg'
    scale = 1.0
elif img == '1920-1930':
    image_name = '1920-1930.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_1920-1930.jpg'
    pixel_y = 'Pixel_y_1920-1930.jpg'
    scale = 1.0
elif img == '1925-1930':
    image_name = '1925-1930.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_1925-1930.jpg'
    pixel_y = 'Pixel_y_1925-1930.jpg'
    scale = 1.0
elif img == '1930':
    image_name = '1930.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_1930.jpg'
    pixel_y = 'Pixel_y_1930.jpg'
    scale = 1.0
elif img == 'center of the settlement kuliang':
    image_name = 'center of the settlement kuliang.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_center of the settlement kuliang.jpg'
    pixel_y = 'Pixel_y_center of the settlement kuliang.jpg'
    scale = 1.0
elif img == 'kuliang hills':
    image_name = 'kuliang hills.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_kuliang hills.jpg'
    pixel_y = 'Pixel_y_kuliang hills.jpg'
    scale = 1.0
elif img == 'kuliang panorama central segment':
    image_name = 'kuliang panorama central segment.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_kuliang panorama central segment.jpg'
    pixel_y = 'Pixel_y_kuliang panorama central segment.jpg'
    scale = 1.0
elif img == 'kuliang Pine Crag road':
    image_name = 'kuliang Pine Crag road.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_kuliang Pine Crag road.jpg'
    pixel_y = 'Pixel_y_kuliang Pine Crag road.jpg'
    scale = 1.0
elif img == 'Siems Siemssen':
    image_name = 'Siems Siemssen.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_Siems Siemssen.jpg'
    pixel_y = 'Pixel_y_Siems Siemssen.jpg'
    scale = 1.0
elif img == 'View Kuliang includes tennis courts':
    image_name = 'View Kuliang includes tennis courts.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_View Kuliang includes tennis courts.jpg'
    pixel_y = 'Pixel_y_View Kuliang includes tennis courts.jpg'
    scale = 1.0
elif img == 'Worley Family-20':
    image_name = 'Worley Family-20.jpg'
    features = 'feature_points_with_annotations.csv'
    pixel_x = 'Pixel_x_Worley Family-20.jpg'
    pixel_y = 'Pixel_y_Worley Family-20.jpg'
    scale = 1.0

image_path = os.path.join("historical photos", image_name)

img = cv2.imread(image_path)
h, w = img.shape[:2]
image_size = (w, h)

points_data = read_points_data(features, pixel_x, pixel_y, scale)

focal_lengths = [90, 100, 120, 150, 180, 210, 240, 300, 360]
sensor_sizes = [
    (102, 127),
    (127, 178),
    (203, 254)
]
known_camera_origin = np.array([7.39410777e+05, 2.88828092e+06, 7.78499264e+02])

filtered_results = estimate_camera_orientation(points_data, focal_lengths, sensor_sizes, image_size)

# 提供交互式选项，让用户选择一个相机位置
selected_index = int(input("Select a camera position (1-5): ")) - 1

if 0 <= selected_index < len(filtered_results):
    selected_result = filtered_results[selected_index]
    optimize_camera_center(selected_result, points_data)
else:
    print("Invalid selection.")
