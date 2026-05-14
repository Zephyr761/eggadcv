from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import StopLineType
from nuplan.planning.nuboard.base.plot_data import MapPoint

from numba import njit, prange
# 编译算子
#from debugipy.cuda_ops import zbuffer_cuda
from cuda_ops import zbuffer_cuda
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

import torch
import numpy as np
import pickle
import cv2
import os
from copy import deepcopy
from pyquaternion import Quaternion
from shapely.geometry import (Polygon, MultiPolygon, MultiPoint, LineString, 
                              MultiLineString)

near_plane = 1e-8
min_polygon_area = 2000
patch_radius = 100


@njit()
def rotate_points_3d(
    points: np.ndarray,
    angle: float,
    axis: int = 2,
    clockwise: bool = True
) -> np.ndarray:


    if clockwise:
        angle = -angle

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.eye(3)  # 初始化为单位矩阵

    if axis == 0: 
        rotation_matrix[1, 1] = cos_a
        rotation_matrix[1, 2] = -sin_a
        rotation_matrix[2, 1] = sin_a
        rotation_matrix[2, 2] = cos_a
    elif axis == 1:  
        rotation_matrix[0, 0] = cos_a
        rotation_matrix[0, 2] = sin_a
        rotation_matrix[2, 0] = -sin_a
        rotation_matrix[2, 2] = cos_a
    elif axis == 2:  
        rotation_matrix[0, 0] = cos_a
        rotation_matrix[0, 1] = -sin_a
        rotation_matrix[1, 0] = sin_a
        rotation_matrix[1, 1] = cos_a

    points = points.astype(np.float32)
    rotation_matrix = rotation_matrix.astype(np.float32)
    rotated_points = np.dot(points[:,:3], rotation_matrix.T)

    return rotated_points


@njit()
def rotate_points_3d_color(
    points: np.ndarray,
    angle: float,
    axis: int = 2,
    clockwise: bool = True
) -> np.ndarray:

    if clockwise:
        angle = -angle

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.eye(3)  # 初始化为单位矩阵
    points = points.astype(np.float32)
    if axis == 0: 
        rotation_matrix[1, 1] = cos_a
        rotation_matrix[1, 2] = -sin_a
        rotation_matrix[2, 1] = sin_a
        rotation_matrix[2, 2] = cos_a
    elif axis == 1:  
        rotation_matrix[0, 0] = cos_a
        rotation_matrix[0, 2] = sin_a
        rotation_matrix[2, 0] = -sin_a
        rotation_matrix[2, 2] = cos_a
    elif axis == 2:  
        rotation_matrix[0, 0] = cos_a
        rotation_matrix[0, 1] = -sin_a
        rotation_matrix[1, 0] = sin_a
        rotation_matrix[1, 1] = cos_a

    xyz = points[:,:3]
    rgb = points[:,3:]
    

    rotation_matrix = rotation_matrix.astype(np.float32)
    rotated_xyz = np.dot(xyz, rotation_matrix.T)
    rotated_points = np.hstack((rotated_xyz, rgb))

    return rotated_points

def cuda_perpare(u, v, z, H, W, clr=None,semantic=None):
    
    u = torch.tensor(u, dtype=torch.int32).cuda()
    v = torch.tensor(v, dtype=torch.int32).cuda()
    z = torch.tensor(z, dtype=torch.float32).cuda()

    if clr is not None:
        clr = torch.tensor(clr, dtype=torch.uint8).cuda()
    if semantic is not None:
        semantic = torch.tensor(semantic, dtype=torch.int32).cuda()

    z_buffer = torch.full((H, W), 1e9, dtype=torch.float32, device='cuda')
    clr_map = torch.zeros((H, W, 3), dtype=torch.uint8, device='cuda')
    semantic_map = torch.zeros((H, W, 3), dtype=torch.uint8, device='cuda') #TODO: hardcode

    return u,v,z,clr,semantic,z_buffer,clr_map,semantic_map

def resize_depth_ignore_invalid(depth_img, target_size):
    mask_invalid = (depth_img <= -300)
    depth_nan = depth_img.astype(np.float32).copy()
    depth_nan[mask_invalid] = np.nan
    
    new_width, new_height = target_size
    h, w = depth_nan.shape

    scale_x = w / float(new_width)
    scale_y = h / float(new_height)

    resized = np.full((new_height, new_width), np.nan, dtype=np.float32)

    for dst_y in range(new_height):
        for dst_x in range(new_width):
            # 对应到原图的位置
            src_x = int(round(dst_x * scale_x))
            src_y = int(round(dst_y * scale_y))
            # 越界保护
            src_x = min(src_x, w-1)
            src_y = min(src_y, h-1)

            # 将原点值复制过来(若为NaN就保持NaN)
            resized[dst_y, dst_x] = depth_nan[src_y, src_x]

    # 3) 插值完成后，再把 NaN 恢复成 -300
    mask_resized_invalid = np.isnan(resized)
    resized[mask_resized_invalid] = -300

    return resized

def downsample_depth_blockwise(depth_img, target_size):
    old_h, old_w = depth_img.shape
    new_h, new_w = old_h // 5, old_w // 5

    # 计算每块在原图中覆盖的高宽
    out = np.full((new_h, new_w), -300, dtype=depth_img.dtype)

    for i in range(new_h):
        for j in range(new_w):
            # 对应到原图中的块
            h_start, h_end = i * 5, (i+1)*5
            w_start, w_end = j * 5, (j+1)*5
            block = depth_img[h_start:h_end, w_start:w_end]

            # 筛掉无真值(-300)，只对剩余的有效值汇聚
            valid_vals = block[block > -200]
            if len(valid_vals) > 0:
                out[i,j] = np.max(valid_vals)

    out = cv2.resize(out, (target_size[1],target_size[0]), interpolation=cv2.INTER_NEAREST)

    return out

def downsample_clr_blockwise(img, target_size):

    old_h, old_w, _ = img.shape
    new_h,new_w = old_h // 4, old_w // 4

    out = np.full((new_h, new_w, 3), 0, dtype=img.dtype)

    for i in range(new_h):
        for j in range(new_w):
            # 对应到原图中的块
            h_start, h_end = i * 4, (i+1)*4
            w_start, w_end = j * 4, (j+1)*4
            block = img[h_start:h_end, w_start:w_end]

            valid_vals = block[np.any(block > 20, axis=-1)]
            if valid_vals.size > 0:
                out[i, j] = np.mean(valid_vals, axis=0)  
            else:
                out[i, j] = [0, 0, 0]  

    out = cv2.resize(out, (target_size[1],target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    return out.astype(np.uint8)  # ✅ 确保数据类型正确




def get_projected_bboxes(
    bboxes,
    labels,
    N_cam,
    im_size,
    final_dim,
    lidar2image,
    object_classes,
    cross_object,
):
    num_classes = len(object_classes)
    is_within_int32 = lambda arr : np.all((arr >= np.iinfo(np.int32).min) & (arr <= np.iinfo(np.int32).max))
    def box_with_in_int32(box):
        for coord in box:
            if not is_within_int32(coord):
                return False
        return True
    
    if len(bboxes) == 0:
        return np.zeros((N_cam ,num_classes, final_dim[0], final_dim[1]), dtype=np.uint8), \
            np.zeros((N_cam, final_dim[0], final_dim[1]), dtype=np.uint8)
    
    scene_poly = Polygon([(0, 0), (0, im_size[1]), (im_size[0], im_size[1]), (im_size[0], 0)])
    ret_mask_list = []
    ret_loss_mask_list = []
    for i in range(N_cam):
        all_cls_mask = np.zeros((num_classes, final_dim[0], final_dim[1]), dtype=np.uint8)
        loss_mask = np.zeros((im_size[1], im_size[0]), dtype=np.uint8)
        for cls_id in range(len(object_classes)):
            cls_mask = labels == cls_id
            cls_boxes = bboxes[cls_mask]
            if len(cls_boxes) == 0:
                continue
            # transform to camera view
            coords = _trans_boxes_to_view(
                bboxes=cls_boxes,
                transform=lidar2image[i],
                proj=True
            )
            # filter out points behind the camera
            indices = np.all(coords[..., 2] > 0, axis=1)
            coords = coords[indices]
            coords = coords[:, :, :2]
            front_index = [(0,5), (1,4)]
            mask = np.zeros((im_size[1], im_size[0]), dtype=np.uint8)
            # mask = deepcopy(np.asarray(data["img"][i]))
            # front_index = [0,1,5,4]
            for box in coords:
                if not box_with_in_int32(box):
                    continue
                if object_classes[cls_id] == cross_object:
                    cv2.line(mask, box[front_index[0][0]].astype(np.int32), box[front_index[0][1]].astype(np.int32),  
                                color=1, thickness=3)
                    cv2.line(mask, box[front_index[1][0]].astype(np.int32), box[front_index[1][1]].astype(np.int32),  
                                color=1, thickness=3)
                # # fill the front size
                # if self.object_classes[cls_id] in \
                #     ("car", "truck", "construction_vehicle", "bus", "trailer"):
                #     front_coords = box[front_index]
                #     cv2.fillPoly(mask, np.int32([front_coords]), color=1)
                # draw line on mask
                for start, end in [
                    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2),
                    (3, 7), (4, 5), (4, 7), (2, 6) , (5, 6), (6, 7),
                ]:
                    cv2.line(mask, box[start].astype(np.int32), box[end].astype(np.int32),  
                                color=1, thickness=3)

            # generate loss mask for certain type of vehicles
            if object_classes[cls_id] == cross_object:
                for box in coords:
                    # get the convex hull of the box
                    multi_point = MultiPoint(box)
                    box = multi_point.convex_hull
                    box = box.intersection(scene_poly)
                    if box.is_empty:
                        continue
                    if box.geom_type == 'Polygon':
                        box = MultiPolygon([box])
                    elif box.geom_type == 'LineString':
                        continue
                    loss_mask = mask_for_polygons(box, loss_mask)
            mask = cv2.resize(mask, (final_dim[1], final_dim[0]))
            all_cls_mask[cls_id] = mask
        loss_mask = cv2.resize(loss_mask, (final_dim[1], final_dim[0]))
        ret_mask_list.append(all_cls_mask)
        ret_loss_mask_list.append(loss_mask)

    return np.stack(ret_mask_list), np.stack(ret_loss_mask_list)

def get_projected_map(
    numap,
    patch_radius,
    N_cam,
    im_size,
    final_dim,
    ego2global,
    sensor2ego_t_list,
    sensor2ego_r_list,
    cam_intrinsics_list,
    polygon_layer_names,
    line_layer_names,
):
    layer_names = list(set(polygon_layer_names) | set(line_layer_names))
    center = Point2D(ego2global[0,3], ego2global[1,3])
    ego2global_t = ego2global[:3, 3]
    ego2global_r = ego2global[:3, :3]
    _nearest_vector_map = numap.get_proximal_map_objects(center, patch_radius, polygon_layer_names)
    # Filter out stop polygons in turn stop
    if SemanticMapLayer.STOP_LINE in _nearest_vector_map:
        stop_polygons = _nearest_vector_map[SemanticMapLayer.STOP_LINE]
        _nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
            stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
        ]
    map_masks_proj = []

    for i in range(N_cam):
        sensor2ego_t = sensor2ego_t_list[i]
        sensor2ego_r = sensor2ego_r_list[i]
        cam_intrinsic = cam_intrinsics_list[i][:3, :3]

        scene_mask = np.zeros(
            (len(polygon_layer_names) + len(line_layer_names), final_dim[0], final_dim[1]), 
            dtype=np.uint8)   
        #############################################
        #           mask for polygon elements       #
        #############################################
        polygons_proj = _get_perspective_polygons(
            _nearest_vector_map, im_size, ego2global_t, ego2global_r, sensor2ego_t, sensor2ego_r, cam_intrinsic, polygon_layer_names
        )
        scene_poly = Polygon([(0, 0), (0, im_size[1]), (im_size[0], im_size[1]), (im_size[0], 0)])
        for layer_name in polygons_proj.keys():
            mask = np.zeros((im_size[1], im_size[0]), dtype=np.uint8)
            for poly in polygons_proj[layer_name]:
                if not poly.is_valid:
                    continue
                poly = poly.intersection(scene_poly)
                if poly.is_empty:
                    continue
                if poly.geom_type == 'Polygon':
                    poly = MultiPolygon([poly])
                elif poly.geom_type == 'LineString':
                    continue
                mask = mask_for_polygons(poly, mask)
            mask = cv2.resize(mask, (final_dim[1], final_dim[0]))
            scene_mask[layer_names.index(layer_name)] = mask

        #############################################
        #           mask for line elements          #
        #############################################
        lines_proj = _get_perspective_lines(
            _nearest_vector_map, im_size, ego2global_t, ego2global_r, sensor2ego_t, sensor2ego_r, cam_intrinsic, line_layer_names
        )
        for layer_name in lines_proj.keys():
            mask = np.zeros((im_size[1], im_size[0]), dtype=np.uint8)

            for line in lines_proj[layer_name]:
                line = line.intersection(scene_poly)
                if line.is_empty:
                    continue
                if line.geom_type == "LineString":
                    line = MultiLineString([line])
                mask_for_lines(line, mask)
            mask = cv2.resize(mask, (final_dim[1], final_dim[0]))
            scene_mask[len(polygon_layer_names) + line_layer_names.index(layer_name)] = mask

        map_masks_proj.append(scene_mask)
    
    return np.stack(map_masks_proj)

def _get_perspective_lines(
    _nearest_vector_map,
    im_size,
    ego2global_t,
    ego2global_r,
    sensor2ego_t,
    sensor2ego_r,
    cam_intrinsic,
    line_layer_names,
):
    ret_lines = {}
    get_perspective_coords = lambda x: _perspective_coords(x, im_size,
            ego2global_t, ego2global_r, sensor2ego_t, sensor2ego_r, cam_intrinsic, False)
    for layer_name in line_layer_names:
        layer_objects = _nearest_vector_map[layer_name]
        lines = []
        for map_obj in layer_objects:
            path = map_obj.left_boundary.discrete_path
            points = [[pose.x, pose.y] for pose in path]
            lines.append(points)
            path = map_obj.right_boundary.discrete_path
            points = [[pose.x, pose.y] for pose in path]
            lines.append(points)
        lines = np.array(lines, dtype=object)
        for coords in lines:
            points = np.array(coords).transpose(1, 0)
            points = np.vstack((points, np.zeros((1, points.shape[1]))))

            points = get_perspective_coords(points)

            if points is None:
                continue

            points = points[:2, :]
            points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
            line_proj = LineString(points)

            ret_lines.setdefault(layer_name, []).append(line_proj)        
    return ret_lines  

def _get_perspective_polygons(
    _nearest_vector_map,
    im_size,
    ego2global_t,
    ego2global_r,
    sensor2ego_t,
    sensor2ego_r,
    cam_intrinsic,
    polygon_layer_names,
):
    ret_polygons = {}
    get_perspective_coords = lambda x: _perspective_coords(x, im_size,
            ego2global_t, ego2global_r, sensor2ego_t, sensor2ego_r, cam_intrinsic, True)
    for layer_name in polygon_layer_names:
        layer_objects = _nearest_vector_map[layer_name]
        for obj in layer_objects:
            polygon: Polygon = obj.polygon
            # Convert polygon nodes to pointcloud with 0 height.
            points = np.array(polygon.exterior.xy)
            points = np.vstack((points, np.zeros((1, points.shape[1]))))
            
            points = get_perspective_coords(points)

            if points is None:
                continue

            points = points[:2, :]
            points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
            polygon_proj = Polygon(points)

            if polygon_proj.area < min_polygon_area:
                continue
        
            ret_polygons.setdefault(layer_name, []).append(polygon_proj)
    return ret_polygons   

def _perspective_coords(
        points,
        im_size,
        ego2global_t,
        ego2global_r,
        sensor2ego_t,
        sensor2ego_r,
        cam_intrinsic,
        is_polygon            
):
    # from global coordiante to lidar coordinate
    # Transform into the ego vehicle frame for the timestamp of the image.
    ego2global_t = np.concatenate([ego2global_t[:2], np.zeros_like(ego2global_t)[:1]], axis=-1)
    points = points - np.array(ego2global_t).reshape((-1, 1))
    points = np.dot(ego2global_r.T, points)
    
    # Transform into the camera.
    points = points - np.array(sensor2ego_t).reshape((-1, 1))
    points = np.dot(Quaternion(sensor2ego_r).rotation_matrix.T, points)

    # Remove points that are partially behind the camera.
    depths = points[2, :]
    behind = depths < near_plane
    if np.all(behind):
        return 

    points = clip_points_behind_camera(points, near_plane, is_polygon)

    # Ignore polygons with less than 3 points after clipping.
    if is_polygon:
        if len(points) == 0 or points.shape[1] < 3:
            return

    points = view_points(points, cam_intrinsic, normalize=True)

    # Skip polygons where all points are outside the image.
    # Leave a margin of 1 pixel for aesthetic reasons.
    inside = np.ones(points.shape[1], dtype=bool)
    inside = np.logical_and(inside, points[0, :] > 1)
    inside = np.logical_and(inside, points[0, :] < im_size[0] - 1)
    inside = np.logical_and(inside, points[1, :] > 1)
    inside = np.logical_and(inside, points[1, :] < im_size[1] - 1)
    if np.all(np.logical_not(inside)):
        return
    return points

def _box_center_shift(bboxes: LiDARInstance3DBoxes, new_center):
    raw_data = bboxes.tensor.numpy()
    new_bboxes = LiDARInstance3DBoxes(
        raw_data, box_dim=raw_data.shape[-1], origin=new_center)
    return new_bboxes

def _trans_boxes_to_view(bboxes, transform, aug_matrix=None, proj=True):
    """2d projection with given transformation.

    Args:
        bboxes (LiDARInstance3DBoxes): bboxes
        transform (np.array): 4x4 matrix
        aug_matrix (np.array, optional): 4x4 matrix. Defaults to None.

    Returns:
        np.array: (N, 8, 3) normlized, where z = 1 or -1
    """
    if len(bboxes) == 0:
        return None

    bboxes_trans = _box_center_shift(bboxes, (0.5, 0.5, 0.5))
    trans = transform
    if aug_matrix is not None:
        aug = aug_matrix
        trans = aug @ trans
    corners = bboxes_trans.corners
    num_bboxes = corners.shape[0]

    coords = np.concatenate(
        [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
    )
    trans = deepcopy(trans).reshape(4, 4)
    coords = coords @ trans.T

    coords = coords.reshape(-1, 4)
    # we do not filter > 0, need to keep sign of z
    if proj:
        z = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= z
        coords[:, 1] /= z
        coords[:, 2] /= np.abs(coords[:, 2])

    coords = coords[..., :3].reshape(-1, 8, 3)
    return coords

def mask_for_lines(lines: MultiLineString, mask: np.ndarray) -> np.ndarray:
    """
    Convert a polygon or multipolygon list to an image mask ndarray.
    :param polygons: List of Shapely polygons to be converted to numpy array.
    :param mask: Canvas where mask will be generated.
    :return: Numpy ndarray polygon mask.
    """
    def int_coords(x):
        # function to round and convert to int
        return np.array(x).round().astype(np.int32)
    lines = [int_coords(line.coords) for line in lines.geoms]
    cv2.polylines(mask, lines, isClosed=False, color=1, thickness=3)
    return mask

def mask_for_polygons(polygons: MultiPolygon, mask: np.ndarray) -> np.ndarray:
        """
        Convert a polygon or multipolygon list to an image mask ndarray.
        :param polygons: List of Shapely polygons to be converted to numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray polygon mask.
        """
        if not polygons:
            return mask

        def int_coords(x):
            # function to round and convert to int
            return np.array(x).round().astype(np.int32)
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons.geoms]
        interiors = [int_coords(pi.coords) for poly in polygons.geoms for pi in poly.interiors]
        cv2.fillPoly(mask, exteriors, 1)
        cv2.fillPoly(mask, interiors, 0)
        return mask

def clip_points_behind_camera(points, near_plane: float, is_polygon):
    points_clipped = []
    # Loop through each line on the polygon.
    # For each line where exactly 1 endpoints is behind the camera, move the point along the line until
    # it hits the near plane of the camera (clipping).
    assert points.shape[0] == 3
    point_count = points.shape[1]
    range_point_count = point_count
    if not is_polygon:
        range_point_count = range_point_count - 1
    for line_1 in range(range_point_count):
        line_2 = (line_1 + 1) % point_count
        point_1 = points[:, line_1]
        point_2 = points[:, line_2]
        z_1 = point_1[2]
        z_2 = point_2[2]

        if z_1 >= near_plane and z_2 >= near_plane:
            # Both points are in front.
            # Add both points unless the first is already added.
            if len(points_clipped) == 0 or all(points_clipped[-1] != point_1):
                points_clipped.append(point_1)
            points_clipped.append(point_2)
        elif z_1 < near_plane and z_2 < near_plane:
            # Both points are in behind.
            # Don't add anything.
            continue
        else:
            # One point is in front, one behind.
            # By convention pointA is behind the camera and pointB in front.
            if z_1 <= z_2:
                point_a = points[:, line_1]
                point_b = points[:, line_2]
            else:
                point_a = points[:, line_2]
                point_b = points[:, line_1]
            z_a = point_a[2]
            z_b = point_b[2]

            # Clip line along near plane.
            pointdiff = point_b - point_a
            alpha = (near_plane - z_b) / (z_a - z_b)
            clipped = point_a + (1 - alpha) * pointdiff
            assert np.abs(clipped[2] - near_plane) < 1e-6

            # Add the first point (if valid and not duplicate), the clipped point and the second point (if valid).
            if z_1 >= near_plane and (len(points_clipped) == 0 or all(points_clipped[-1] != point_1)):
                points_clipped.append(point_1)
            points_clipped.append(clipped)
            if z_2 >= near_plane:
                points_clipped.append(point_2)

    points_clipped = np.array(points_clipped).transpose()
    return points_clipped

def get_ds_map(ego_pose, ego2global, track_token, gt_boxes, gt_labels, clr_scene, actor_root, lidar2image, static_actors):
    ego_pose = np.array([ego_pose.x, ego_pose.y])
    ego2global_t = ego2global[:3, 3]
    ego2global_r = ego2global[:3, :3]
    # around dense points
    radius = int(200)
    mask = (clr_scene[:, 0] > ego_pose[0] - radius) & (clr_scene[:, 0] < ego_pose[0] + radius) & \
    (clr_scene[:, 1] > ego_pose[1] - radius) & (clr_scene[:, 1] < ego_pose[1] + radius)
    clr_pts = clr_scene[mask]
    pts = deepcopy(clr_pts[:,:3])
    # distances_clr = np.sqrt((clr_scene[:, 0] - ego_pose[0])**2 + ((clr_scene[:, 1] - ego_pose[1])**2))
    # clr_pts = clr_scene[distances_clr <= radius]
    # pts = deepcopy(clr_pts[:,:3])

    #global -> ego
    pts = pts[:,:3] @ np.linalg.inv(ego2global_r).T - ego2global_t @ np.linalg.inv(ego2global_r).T
    pts =  np.concatenate([pts, np.zeros((pts.shape[0], 1))], axis=1)

    clr_pts[:,:3] = clr_pts[:,:3] @ np.linalg.inv(ego2global_r).T - ego2global_t @ np.linalg.inv(ego2global_r).T

    # add actor wo clr    
    actor_list = []

    for idx, box in enumerate(gt_boxes):
        actor_path = os.path.join(actor_root, f'{track_token[idx]}.npy')
        actor = np.load(actor_path) if os.path.exists(actor_path) else None
        if actor is not None and actor.shape[0] > 20000:
            if gt_labels[idx] == 0:
                actor = rotate_points_3d(actor[:,:3], box[6], axis=2, clockwise=False)
                translation_vector = np.array([box[0], box[1], box[2]])
                actor = actor + translation_vector
                actor = np.concatenate([actor, np.ones((actor.shape[0], 1))], axis=1)
                actor_list.append(actor)
                # pts = np.concatenate([pts, actor], axis=0)
            elif gt_labels[idx] == 1:
                actor = rotate_points_3d(actor[:,:3], box[6], axis=2, clockwise=False)
                translation_vector = np.array([box[0], box[1], box[2]])
                actor = actor + translation_vector
                semantic = np.full((actor.shape[0], 1), 2)
                actor = np.concatenate([actor, semantic], axis=1)
                actor_list.append(actor)
                # pts = np.concatenate([pts, actor], axis=0)
            elif gt_labels[idx] == 2:
                actor = rotate_points_3d(actor[:,:3], box[6], axis=2, clockwise=False)
                translation_vector = np.array([box[0], box[1], box[2]])
                actor = actor + translation_vector
                semantic = np.full((actor.shape[0], 1), 3)
                actor = np.concatenate([actor, semantic], axis=1)
                actor_list.append(actor)
                # pts = np.concatenate([pts, actor], axis=0)
        else:
            if gt_labels[idx] == 0:
                if box[5] < 1.8:
                    actor = rotate_points_3d(static_actors['sedan'], box[6], axis=2, clockwise=False)
                    translation_vector = np.array([box[0], box[1], box[2]])
                    actor = actor + translation_vector
                    actor = np.concatenate([actor, np.ones((actor.shape[0], 1))], axis=1)
                    actor_list.append(actor)
                    # pts = np.concatenate([pts, actor], axis=0)

                elif box[5] < 2.05:
                    actor = rotate_points_3d(static_actors['suv'], box[6], axis=2, clockwise=False)
                    translation_vector = np.array([box[0], box[1], box[2]])
                    actor = actor + translation_vector
                    actor = np.concatenate([actor, np.ones((actor.shape[0], 1))], axis=1)
                    actor_list.append(actor)
                    # pts = np.concatenate([pts, actor], axis=0)

                elif box[5] < 2.7:
                    actor = rotate_points_3d(static_actors['pickup'], box[6], axis=2, clockwise=False)
                    translation_vector = np.array([box[0], box[1], box[2]])
                    actor = actor + translation_vector
                    actor = np.concatenate([actor, np.ones((actor.shape[0], 1))], axis=1)
                    actor_list.append(actor)
                    # pts = np.concatenate([pts, actor], axis=0)

            elif gt_labels[idx] == 1:
                actor = rotate_points_3d(static_actors['ped'], box[6], axis=2, clockwise=False)
                translation_vector = np.array([box[0], box[1], box[2]])
                actor = actor + translation_vector
                semantic = np.full((actor.shape[0], 1), 2)
                actor = np.concatenate([actor, semantic], axis=1)
                actor_list.append(actor)
                # pts = np.concatenate([pts, actor], axis=0)
            
            elif gt_labels[idx] == 2:
                actor = rotate_points_3d(static_actors['bike'], box[6], axis=2, clockwise=False)
                translation_vector = np.array([box[0], box[1], box[2]])
                actor = actor + translation_vector
                semantic = np.full((actor.shape[0], 1), 3)
                actor = np.concatenate([actor, semantic], axis=1)
                actor_list.append(actor)
                # pts = np.concatenate([pts, actor], axis=0)

    pts = np.concatenate([pts, np.concatenate(actor_list, axis=0)], axis=0)

    # t3 = time.time()
    # print(f"[TIMER] Add actor: {t3 - t2:.3f}s")

    actor_list = []
    # add actor with clr
    for idx, box in enumerate(gt_boxes):
        actor_path = os.path.join(actor_root, f'track_actor/{track_token[idx]}.npy')
        actor = np.load(actor_path) if os.path.exists(actor_path) else None
        if actor is not None:
            actor = np.load(os.path.join(actor_root, f'track_actor/{track_token[idx]}.npy'))
            actor = rotate_points_3d_color(actor, box[6], axis=2, clockwise=False)
            translation_vector = np.array([box[0], box[1], box[2]])
            actor[:,:3] += translation_vector
            actor_list.append(actor)
            
    clr_pts = np.concatenate([clr_pts, np.concatenate(actor_list, axis=0)], axis=0)


    xyz = pts[:, :3]  # 点云的 (x, y, z) 坐标
    xyz_clr = clr_pts[:, :3]
    semantics = pts[:, 3]  # 点云的语义类别

    xyz = np.concatenate(
            [xyz, np.ones((xyz.shape[0], 1))], axis=-1
        )
    xyz_clr = np.concatenate(
            [xyz_clr, np.ones((xyz_clr.shape[0], 1))], axis=-1
        )

    ncam_depth = []
    ncam_clr = []
    ncam_sem = []

    #ego->camera
    for idx, cam in enumerate(['CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_B0','CAM_R2','CAM_R1','CAM_R0']):
        trans = lidar2image[idx]

        points_2d = xyz @ trans.T
        points_clr_2d = xyz_clr @ trans.T

        valid_idx = points_2d[:, 2] > 1e-5
        valid_clr_idx = points_clr_2d[:, 2] > 1e-5
        points_2d = points_2d[valid_idx]
        semantic = semantics[valid_idx]
        points_clr_2d = points_clr_2d[valid_clr_idx]
        rgb_pts = clr_pts[valid_clr_idx]

        points_2d[:, 2] = np.clip(points_2d[:, 2], a_min=1e-5, a_max=1e5)
        points_2d[:, 0] /= points_2d[:, 2]  # u = fx * x / z + cx
        points_2d[:, 1] /= points_2d[:, 2]  # v = fy * y / z + cy

        points_clr_2d[:, 2] = np.clip(points_clr_2d[:, 2], a_min=1e-5, a_max=1e5)
        points_clr_2d[:, 0] /= points_clr_2d[:, 2]  # u = fx * x / z + cx
        points_clr_2d[:, 1] /= points_clr_2d[:, 2]  # v = fy * y / z + cy

        u = points_2d[:, 0].astype(np.int16)
        v = points_2d[:, 1].astype(np.int16)
        z = points_2d[:, 2] 

        u_clr = points_clr_2d[:, 0].astype(np.int16)
        v_clr = points_clr_2d[:, 1].astype(np.int16)
        z_clr = points_clr_2d[:, 2]

        r, g, b = rgb_pts[:, 3], rgb_pts[:, 4], rgb_pts[:, 5]

        height, width = (1080, 1920)
        depth_map = np.full((height, width), np.inf, dtype=np.float32)  # 初始化为无穷大，表示无深度值
        semantic_map = np.zeros((height, width, 3), dtype=np.uint8)  #TODO: 应改成目标类别
        clr_map = np.zeros((height, width, 3), dtype=np.uint8)

        valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid_clr_mask = (u_clr >= 0) & (u_clr < width) & (v_clr >= 0) & (v_clr < height)
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z[valid_mask]
        semantic_valid = semantic[valid_mask]
        
        u_clr_valid = u_clr[valid_clr_mask]
        v_clr_valid = v_clr[valid_clr_mask]
        z_clr_valid = z_clr[valid_clr_mask]
        r_valid = r[valid_clr_mask]
        g_valid = g[valid_clr_mask]
        b_valid = b[valid_clr_mask]

        clr = np.stack([r_valid, g_valid, b_valid], axis=1)
        target_size = (224,400)
        # t_cuda_2 = time.time()
        # print(f"[TIMER] Cam{idx} - prepare (clr): {t_cuda_2 - t_cuda_1:.3f}s")
        #clr
        u_clr_valid, v_clr_valid, z_clr_valid, clr, _, z_buffer, clr_map, semantic_map = cuda_perpare(u_clr_valid,v_clr_valid,z_clr_valid,height,width,clr)
        zbuffer_cuda.zbuffer(u_clr_valid, v_clr_valid, z_clr_valid,
                            None, clr,
                            z_buffer, semantic_map, clr_map,
                            height, width, 0, 0
                        ) #-2参数是光栅化半径 6*6  获取clrmap
        clr_map = clr_map.cpu().numpy()

        clr_map = downsample_clr_blockwise(clr_map, target_size)
        ncam_clr.append(clr_map)

        #depth, sem
        u_valid, v_valid, z_valid, _, semantic, z_buffer, clr_map, semantic_map = cuda_perpare(u_valid,v_valid,z_valid,height,width,semantic=semantic_valid)
        zbuffer_cuda.zbuffer(u_valid, v_valid, z_valid,
                semantic, None,
                z_buffer, semantic_map, clr_map,
                height, width, 0, 3
            ) 
            
        depth_map = z_buffer.cpu().numpy()
        depth_map[depth_map == 1e9] = -300
        semantic_map = semantic_map.cpu().numpy()

        semantic_map = cv2.resize(semantic_map, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        depth_map = downsample_depth_blockwise(depth_map, target_size)


        # 逐点更新
        # for i in range(len(u_valid)):
        #     u_idx, v_idx = u_valid[i], v_valid[i]
        #     if z_valid[i] < depth_map[v_idx, u_idx]:
        #         category = semantic_valid[i]
        #         if 0 <= category <= 2:  # 确保 category 在 0,1,2 之间
        #             semantic_map[v_idx, u_idx, int(category)] = 1 
        #             depth_map[v_idx, u_idx] = z_valid[i]  # 更新深度

        # z_buffer = np.full((height, width), np.inf)
        # for i in range(len(u_clr_valid)):
        #     u, v = int(u_clr_valid[i]), int(v_clr_valid[i])
        #     z = z_clr_valid[i]  # 当前点的深度（越小越近）
        #     color = np.array([r_valid[i], g_valid[i], b_valid[i]], dtype=np.uint8)
        #     if z < z_buffer[v, u]:
        #         z_buffer[v, u] = z  # 更新 z-buffer
        #         clr_map[v, u] = color  # 更新颜色
                
        # depth_map[depth_map == np.inf] = -300
        # target_size = (224,400)
        # # 单独保存sem map
        # semantic_map = cv2.resize(semantic_map, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        # # depth_map = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_NEAREST)
        # # clr_map = cv2.resize(clr_map, target_size, interpolation=cv2.INTER_NEAREST)
        # depth_map = downsample_depth_blockwise(depth_map, target_size)
        # clr_map = downsample_clr_blockwise(clr_map, target_size)

        ncam_depth.append(depth_map)
        ncam_sem.append(semantic_map)
        # for i in range(1,4): # category id set in dataset
        #     sem_mask = (semantic_map == i)
        #     data["gt_projected_boxes"][idx][i-1][sem_mask] = 1
    sem_map = np.stack(ncam_sem).transpose(0,3,1,2)
    clr_map = np.expand_dims(np.stack(ncam_clr), axis=1)
    depth_map = np.expand_dims(np.stack(ncam_depth), axis=1)

    return sem_map, clr_map, depth_map


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points
