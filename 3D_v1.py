import numpy as np
import plotly.graph_objects as go
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt


# 读取DEM数据
def read_dem(file_path):
    with rasterio.open(file_path) as dataset:
        dem_data = dataset.read(1)  # 读取第一波段
        transform = dataset.transform
        x_min = transform[2]
        y_min = transform[5]
        cell_size = transform[0]
    return dem_data, x_min, y_min, cell_size


# 获取DEM数据中的高程值
def get_elevation(dem_data, lon, lat, x_min, y_min, cell_size):
    # 计算行和列索引
    col = int((lon - x_min) / cell_size)
    row = int((y_min - lat) / cell_size)
    if row < 0 or row >= dem_data.shape[0] or col < 0 or col >= dem_data.shape[1]:
        raise ValueError(f"Coordinates ({lat}, {lon}) are out of the DEM bounds")
    return dem_data[row, col]


# 将UTM坐标转换为WGS84坐标
def convert_utm_to_wgs84(utm_coordinates, zone_number, northern_hemisphere=True):
    proj_utm = f"epsg:326{zone_number}" if northern_hemisphere else f"epsg:327{zone_number}"
    proj_wgs84 = "epsg:4326"

    transformer = Transformer.from_crs(proj_utm, proj_wgs84)

    wgs84_coordinates = []
    for easting, northing, elevation in utm_coordinates:
        lat, lon = transformer.transform(easting, northing)
        wgs84_coordinates.append([lon, lat, elevation])
    return wgs84_coordinates


# 生成区域内部的所有点
def generate_internal_points(polygon, cell_size):
    min_x, min_y, max_x, max_y = polygon.bounds
    x_coords = np.arange(min_x, max_x, cell_size)
    y_coords = np.arange(min_y, max_y, cell_size)
    internal_points = []
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            if polygon.contains(point):
                internal_points.append((x, y))
    return internal_points


# UTM坐标数据
utm_coordinates = [
    [738949.8403, 2888807.572, 715.904285],
    [739270.2181, 2888392.707, 708.7734616],
    [739285.6864, 2888403.652, 707.3754047],
    [739343.4971, 2888425.792, 709.379681],
    [739204.0719, 2889154.157, 689.2484],
    [739190.1219, 2889096.59, 693.9828318],
    [739153.4228, 2889083.295, 700.7911165],
    [739142.2623, 2889061.576, 701.1656352],
    [739130.6916, 2889063.589, 703.1555533],
    [739134.4604, 2889016.755, 706.1689749],
    [739137.339, 2888956.638, 707.0927378],
    [739121.1237, 2888915.814, 708.8607068],
    [739112.0061, 2888913.501, 710.9432642],
    [739073.6059, 2888962.342, 721.3875213],
    [739042.0231, 2888965.967, 725.5208791],
    [738991.684, 2888867.475, 725.11944],
    [738974.4293, 2888854.817, 723.7853541],
    [738976.0111, 2888841.201, 722.1436833],
    [738966.5145, 2888841.897, 722.5583878],
    [738959.4964, 2888824.11, 719.028915],
    [738949.1725, 2888817.469, 718.6750427],
]

# 假设你的UTM坐标在第50N区
zone_number = 50
wgs84_coordinates = convert_utm_to_wgs84(utm_coordinates, zone_number)

# 打印转换后的WGS84坐标
print("转换后的WGS84坐标:")
for utm, wgs84 in zip(utm_coordinates, wgs84_coordinates):
    print(f"UTM: {utm} -> WGS84: {wgs84}")

# 创建Polygon对象
polygon = Polygon([(lon, lat) for lon, lat, _ in wgs84_coordinates])

# 输出多边形边界坐标
polygon_boundary = list(polygon.exterior.coords)
print("多边形边界坐标:")
for coord in polygon_boundary:
    print(coord)

# 绘制多边形和UTM坐标点
fig, ax = plt.subplots()
x, y = polygon.exterior.xy
ax.plot(x, y, label='Polygon Boundary')
utm_lon = [coord[0] for coord in wgs84_coordinates]
utm_lat = [coord[1] for coord in wgs84_coordinates]
ax.scatter(utm_lon, utm_lat, color='red', label='UTM Coordinates')

# DEM文件路径（相对路径）
dem_file_path = 'dem_data.tif'

# 读取DEM数据
dem_data, x_min, y_min, cell_size = read_dem(dem_file_path)

# 打印DEM数据元数据
print(f"DEM x_min: {x_min}, y_min: {y_min}, cell_size: {cell_size}")
print(f"DEM data shape: {dem_data.shape}")

# 生成更密集的区域内部点
dense_cell_size = cell_size / 10  # 使内部点更加密集
internal_points = generate_internal_points(polygon, dense_cell_size)

# 输出区域内部点坐标
print("区域内部点坐标:")
for point in internal_points:
    print(point)

# 提取区域内部点的高程值
internal_elevations = []
for lon, lat in internal_points:
    try:
        elevation = get_elevation(dem_data, lon, lat, x_min, y_min, cell_size)
        internal_elevations.append((lon, lat, elevation))
    except ValueError as e:
        print(e)
        continue

# 输出区域内部点的高程值
print("区域内部点的高程值:")
for coord in internal_elevations:
    print(coord)

# 提取X, Y, Z坐标
x = [point[0] for point in internal_elevations]
y = [point[1] for point in internal_elevations]
z = [point[2] for point in internal_elevations]

# 绘制区域内部点
internal_lon = [point[0] for point in internal_points]
internal_lat = [point[1] for point in internal_points]
ax.scatter(internal_lon, internal_lat, color='blue', label='Internal Points')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()
plt.title('Polygon, UTM Coordinates, and Internal Points')
plt.show()

# 创建三维图形
fig = go.Figure()

# 调整高程数据的比例以保持单位一致
z_scale = 111000  # 1度约等于111km

# 创建颜色渐变
colorscale = 'Viridis'
color = [point[2] for point in internal_elevations]

# 添加多边形
fig.add_trace(go.Mesh3d(x=x, y=y, z=[z_i / z_scale for z_i in z], intensity=color, colorscale=colorscale,
                        colorbar_title='Elevation', opacity=0.50))

# 添加原始UTM坐标转换后的WGS84坐标点
utm_lon = [coord[0] for coord in wgs84_coordinates]
utm_lat = [coord[1] for coord in wgs84_coordinates]
utm_elevation = [coord[2] for coord in wgs84_coordinates]

fig.add_trace(go.Scatter3d(x=utm_lon, y=utm_lat, z=[elev / z_scale for elev in utm_elevation],
                           mode='markers', marker=dict(size=5, color='red'),
                           name='UTM Coordinates'))

# 设置图形参数
fig.update_layout(scene=dict(
    xaxis_title='Longitude',
    yaxis_title='Latitude',
    zaxis_title='Elevation (scaled)',
    aspectmode='data'),
    width=800,
    height=800,
    title='3D Terrain Visualization with UTM Coordinates'
)

# 显示图形
fig.show()