"""
DreamForge版本的NuScenes数据加载器

融合特性：
1. DreamForge原版：motion frames机制（9帧->3个VAE latent）、vector map、layout canvas
2. DWM框架：相机/ego变换、点云处理

Motion Frames机制：
- ref_length=9帧（motion frames用于运动建模）
- video_length=8帧（实际生成的视频帧）
- VAE编码后9帧→3个latent（通过temporal_stride=3）

Layout Canvas：
- 包含vector map投影（3通道）+ 3D boxes投影（10通道）
- 总共13通道的投影条件
"""

import dwm.common
import dwm.datasets.common
import dwm.datasets.nuscenes_common

import os
import cv2
import time
import torch
import einops
import fsspec
import json
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFile
from pyquaternion import Quaternion

import torchvision.transforms.functional
from torchvision.transforms.functional import resize as tv_resize, to_tensor as tv_to_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True
cv2.setNumThreads(0)


# ============================================================================
# 辅助函数：Motion Frames处理（来自DreamForge）
# ============================================================================

def _sample_motion_frames(scene_frames, start_idx, ref_length=9, candidate_length=8):
    """
    DreamForge的motion frames采样逻辑

    从clip起点之前的前candidate_length帧中采样ref_length帧作为motion frames。

    Args:
        scene_frames: 场景中的所有帧索引列表
        start_idx: 当前clip的起始位置
        ref_length: motion frames的数量（默认9，VAE编码后得到3个latent）
        candidate_length: 候选帧的范围（默认8）

    Returns:
        ref_idx: motion frames的索引列表
    """
    if start_idx == 0:
        # 第一个clip：重复第一帧
        return [0] * ref_length
    else:
        # 从前candidate_length帧中采样
        candidate_range = list(range(max(start_idx - candidate_length, 0), start_idx))

        if len(candidate_range) < ref_length:
            # 候选不足，允许重复采样（数据增强）
            import random
            ref_idx = sorted(random.choices(candidate_range, k=ref_length))
        else:
            # 候选充足，进行有放回采样（因为candidate_length < ref_length）
            import random
            ref_idx = sorted(random.choices(candidate_range, k=ref_length))

    return ref_idx


def obtain_next2top(first, current, epsilon=1e-6, v2=True):
    """
    计算从first到current的相对位姿变换

    与DreamForge原始的obtain_next2top完全一致
    参考: dreamforgedit/datasets/nuscenes_map_dataset_t.py 第37-91行

    Args:
        first: 第一帧的sample_data信息
        current: 当前帧的sample_data信息
        epsilon: 小阈值，用于过滤微小值
        v2: True返回inverse形式（A @ point_lidar -> point_next），False返回forward形式

    Returns:
        next2lidar: 4x4变换矩阵
    """
    # first: lidar->ego->global
    l2e_r = first.get("lidar2ego_rotation", [0, 0, 0, 1])
    l2e_t = first.get("lidar2ego_translation", [0, 0, 0])
    e2g_r = first.get("ego2global_rotation", [0, 0, 0, 1])
    e2g_t = first.get("ego2global_translation", [0, 0, 0])

    # current: lidar->ego->global
    l2e_r_s = current.get("lidar2ego_rotation", [0, 0, 0, 1])
    l2e_t_s = current.get("lidar2ego_translation", [0, 0, 0])
    e2g_r_s = current.get("ego2global_rotation", [0, 0, 0, 1])
    e2g_t_s = current.get("ego2global_translation", [0, 0, 0])

    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )

    next2lidar_rotation = R.T  # points @ R.T + T
    next2lidar_translation = T

    if v2:
        # inverse, point trans from lidar to next
        _R = np.concatenate([next2lidar_rotation.T, np.array(
            [[0., ] * 3], dtype=T.dtype)], axis=0)
        _T = -next2lidar_rotation.T @ next2lidar_translation
        _T = np.concatenate(
            [_T[..., np.newaxis], np.array([[1.]], dtype=T.dtype)], axis=0)
        # shape like:
        # | R T |
        # | 0 1 |
        # A @ point lidar -> point next
        next2lidar = np.concatenate([_R, _T], axis=1)
    else:
        _R = np.concatenate(
            [next2lidar_rotation, np.array([[0.,]] * 3, dtype=T.dtype)], axis=1)
        _T = np.concatenate(
            [next2lidar_translation, np.array([1.], dtype=T.dtype)], axis=0)
        # shape like:
        # | R 0 |
        # | T 1 |.T
        next2lidar = np.concatenate(
            [_R, _T[np.newaxis, ...]], axis=0,
        ).T  # A @ [points, 1].T

    if epsilon is not None:
        next2lidar[np.abs(next2lidar) < epsilon] = 0.

    return next2lidar


# 保留旧的函数名作为别名（向后兼容）
_compute_relative_pose = obtain_next2top


# ============================================================================
# Vector Map投影函数（DreamForge特性）
# ============================================================================

def project_lines_to_image(lines_coords, line_labels, camera_intrinsic, camera2ego,
                          image_size=(900, 1600), num_classes=3):
    """
    将vector map投影到图像平面

    Args:
        lines_coords: List[np.ndarray]，每条线是(N, 2)的BEV坐标
        line_labels: np.ndarray，每条线的类别
        camera_intrinsic: (3, 3)内参
        camera2ego: (4, 4)相机到ego的变换
        image_size: (H, W)
        num_classes: 地图类别数

    Returns:
        canvas: (H, W, num_classes) one-hot编码地图
    """
    H, W = image_size
    canvas = np.zeros((H, W, num_classes), dtype=np.float32)

    # ego到camera的变换
    ego2camera = np.linalg.inv(camera2ego)

    for line_pts, label in zip(lines_coords, line_labels):
        if len(line_pts) < 2:
            continue

        # 将2D点扩展到3D（z=0，地面）
        pts_3d = np.concatenate([line_pts, np.zeros((len(line_pts), 1))], axis=1)  # (N, 3)
        pts_3d_homo = np.concatenate([pts_3d, np.ones((len(pts_3d), 1))], axis=1).T  # (4, N)

        # 变换到相机坐标系
        pts_cam = ego2camera @ pts_3d_homo  # (4, N)
        pts_cam = pts_cam[:3, :]  # (3, N)

        # 过滤相机后方的点
        valid_mask = pts_cam[2, :] > 0.1
        if not np.any(valid_mask):
            continue

        pts_cam = pts_cam[:, valid_mask]

        # 投影到图像
        pts_img = camera_intrinsic @ pts_cam  # (3, N)
        pts_img = pts_img[:2, :] / (pts_img[2:3, :] + 1e-6)  # (2, N)
        pts_img = pts_img.T.astype(np.int32)  # (N, 2)

        # 过滤越界点
        valid = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < W) & \
                (pts_img[:, 1] >= 0) & (pts_img[:, 1] < H)
        pts_img = pts_img[valid]

        if len(pts_img) < 2:
            continue

        # 绘制线段
        for i in range(len(pts_img) - 1):
            cv2.line(canvas[:, :, int(label)],
                    tuple(pts_img[i]), tuple(pts_img[i + 1]),
                    1.0, thickness=2)

    return canvas


def project_boxes_to_image(boxes_3d, box_labels, lidar2image, image_size=(900, 1600), num_classes=10):
    """
    将3D box投影到图像平面

    Args:
        boxes_3d: (N, 9) [cx, cy, cz, l, w, h, yaw, vx, vy] 或类似格式
        box_labels: (N,) 类别
        lidar2image: (4, 4) lidar到图像的投影
        image_size: (H, W)
        num_classes: 对象类别数

    Returns:
        canvas: (H, W, num_classes) one-hot编码box
    """
    H, W = image_size
    canvas = np.zeros((H, W, num_classes), dtype=np.float32)

    # 3D box角点模板
    corner_template = np.array([
        [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]
    ]).T  # (3, 8)

    for box, label in zip(boxes_3d, box_labels):
        if label < 0 or label >= num_classes:
            continue

        # 解析box参数 - 支持多种格式
        if len(box) >= 7:
            cx, cy, cz, l, w, h, yaw = box[:7]

            # 构建角点
            corners = corner_template * np.array([[l], [w], [h]])  # (3, 8)

            # 旋转
            rot_mat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            corners = rot_mat @ corners

            # 平移
            corners += np.array([[cx], [cy], [cz]])
        else:
            continue

        # 齐次坐标
        corners_homo = np.vstack([corners, np.ones((1, 8))])  # (4, 8)

        # 投影
        pts_img = lidar2image @ corners_homo  # (4, 8)
        pts_img = pts_img[:2, :] / (pts_img[2:3, :] + 1e-6)  # (2, 8)
        pts_img = pts_img.T.astype(np.int32)  # (8, 2)

        # 绘制边
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),  # 底面
            (4, 5), (4, 6), (5, 7), (6, 7),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 竖边
        ]

        for i, j in edges:
            pt1, pt2 = pts_img[i], pts_img[j]
            if (0 <= pt1[0] < W and 0 <= pt1[1] < H) or \
               (0 <= pt2[0] < W and 0 <= pt2[1] < H):
                cv2.line(canvas[:, :, int(label)],
                        tuple(pt1), tuple(pt2),
                        1.0, thickness=2)

    return canvas


def generate_layout_canvas(segment, map_classes=3, object_classes=10, image_size=(900, 1600)):
    """
    为每个帧生成layout canvas（投影的map + box）

    Args:
        segment: List[List[dict]] - [T, Views] 的sample_data
        map_classes: 地图类别数（默认3：divider, ped_crossing, boundary）
        object_classes: 对象类别数（默认10）
        image_size: 图像尺寸(H, W)

    Returns:
        layout_list: List[torch.Tensor] - 每帧的layout canvas [V, 13, H, W]
    """
    layout_list = []

    for frame_sds in segment:
        frame_layouts = []

        for sd in frame_sds:
            # 默认空canvas
            canvas = np.zeros((image_size[0], image_size[1], map_classes + object_classes), dtype=np.float32)

            # 获取相机信息
            calib = sd.get("calibrated_sensor", {})
            camera_intrinsic = calib.get("camera_intrinsic")
            camera2ego = calib.get("camera2ego")

            if camera_intrinsic is not None and camera2ego is not None:
                # 这里简化处理，实际需要从hdmap和3dbox提取数据
                # TODO: 实现真实的vector map和3D boxes投影
                pass

            frame_layouts.append(torch.from_numpy(canvas).permute(2, 0, 1))  # [13, H, W]

        layout_list.append(torch.stack(frame_layouts, dim=0))  # [V, 13, H, W]

    return layout_list  # List[T, V, 13, H, W]


# ============================================================================
# 辅助函数：缓存相关
# ============================================================================

def _try_open_png(p):
    """尝试打开PNG文件"""
    try:
        with Image.open(p) as im:
            im.load()
            return im.convert("RGB")
    except Exception:
        return None


def _safe_save_png(pil_img, p):
    """安全保存PNG"""
    tmp = p + ".tmp"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pil_img.save(tmp, format="PNG", optimize=True)
    os.replace(tmp, p)


def _png_path(cache_root, subdir, token):
    """生成PNG缓存路径"""
    return os.path.join(cache_root, subdir, f"{token}.png")


# ============================================================================
# 主数据集类
# ============================================================================

class MotionDatasetDreamForge(torch.utils.data.Dataset):
    """
    DreamForge版本的NuScenes数据加载器

    融合特性：
    1. DreamForge原版：motion frames机制、vector map、layout canvas
    3. DWM框架：投影深度图、点云处理

    Motion Frames机制：
    - ref_length=9帧（motion frames用于运动建模）
    - video_length=8帧（实际生成的视频帧）
    - VAE编码后9帧→3个latent（通过temporal_stride=3）

    Layout Canvas：
    - 包含vector map投影（3通道）+ 3D boxes投影（10通道）
    - 总共13通道的投影条件

    Args:
        fs: fsspec文件系统
        dataset_name: 数据集名称（如"v1.0-trainval"）
        sequence_length: 视频序列长度（用于枚举clips）
        fps_stride_tuples: [(fps, stride), ...]
        split: 数据划分（"train"/"val"）
        sensor_channels: 相机列表

        # DreamForge特性
        video_length: 实际生成的帧数（默认8）
        ref_length: motion frames数量（默认9，VAE编码后3个latent）
        candidate_length: motion候选范围（默认8）
        vae_temporal_stride: VAE时间步长（默认3，9帧→3latent）

        # Map特性
        map_classes: 地图类别
        object_classes: 对象类别
        generate_layout: 是否生成layout canvas
        layout_canvas_size: layout canvas尺寸

        # 其他
        cache_root: 缓存目录
        enable_camera_transforms: 是否加载相机变换
        keyframe_only: 仅使用keyframe
    """

    table_names = [
        "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
        "sample", "sample_annotation", "sample_data", "scene", "sensor"
    ]

    prune_table_plan = [
        ("sample", "scene_token", "scene"),
        ("sample_data", "sample_token", "sample"),
        ("sample_annotation", "sample_token", "sample")
    ]

    index_names = [
        "calibrated_sensor.token", "category.token", "ego_pose.token",
        "instance.token", "log.token", "map.token", "sample.token",
        "sample_data.sample_token", "sample_data.token",
        "sample_annotation.sample_token", "sample_annotation.token",
        "scene.token", "sensor.token"
    ]

    serialized_table_names = [
        "sample", "sample_annotation", "sample_data", "scene"
    ]

    def __init__(
        self,
        fs: fsspec.AbstractFileSystem,
        dataset_name: str,
        sequence_length: int,
        fps_stride_tuples: list,
        split=None,
        sensor_channels: list = ["CAM_FRONT", "CAM_BACK", "CAM_BACK_LEFT",
                                 "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"],
        keyframe_only: bool = False,
        enable_synchronization_check: bool = True,

        # DreamForge特性
        video_length: int = 8,
        ref_length: int = 9,
        candidate_length: int = 8,
        vae_temporal_stride: int = 3,
        start_on_keyframe: bool = False,

        # Map特性
        map_classes: list = ['divider', 'ped_crossing', 'boundary'],
        object_classes: list = None,
        generate_layout: bool = False,
        layout_canvas_size: tuple = (256, 448),

        # 其他
        cache_root: str = "dataset/nus_cache",
        enable_camera_transforms: bool = True,
        enable_ego_transforms: bool = False,
        enable_sample_data: bool = False,
        _3dbox_image_settings: dict = None,
        hdmap_image_settings: dict = None,
        image_description_settings: dict = None,
        stub_key_data_dict: dict = None,
    ):
        # 导入基础MotionDataset以复用方法
        from dwm.datasets.nuscenes import MotionDataset

        self.fs = fs
        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.keyframe_only = keyframe_only
        self.enable_synchronization_check = enable_synchronization_check

        # DreamForge参数
        self.video_length = video_length
        self.ref_length = ref_length
        self.candidate_length = candidate_length
        self.vae_temporal_stride = vae_temporal_stride
        self.start_on_keyframe = start_on_keyframe

        # Map参数
        self.map_classes = map_classes
        self.object_classes = object_classes or [
            'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
            'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
        ]
        self.generate_layout = generate_layout
        self.layout_canvas_size = layout_canvas_size

        # 其他
        self.cache_root = cache_root
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self.enable_sample_data = enable_sample_data
        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self.image_description_settings = image_description_settings
        self.stub_key_data_dict = stub_key_data_dict

        # 加载数据表
        tables, self.indices = MotionDataset.load_tables(
            fs, dataset_name, self.table_names,
            self.prune_table_plan, self.index_names, split
        )

        # 合并ego_pose到sample_data（减少内存）
        if "ego_pose" in tables:
            for i in tables["sample_data"]:
                pose = MotionDataset.query(
                    tables, self.indices, "ego_pose", i["ego_pose_token"]
                )
                i.update({k: v for k, v in pose.items() if k not in i})
            tables.pop("ego_pose")
            self.indices.pop("ego_pose.token")

        # 构建场景-通道-帧的层次结构
        key_filter = (lambda i: i["is_key_frame"]) if keyframe_only else (lambda _: True)

        scene_channel_sample_data = [
            (scene, [
                sorted([
                    sample_data
                    for sample in MotionDataset.get_scene_samples(tables, self.indices, scene)
                    for sample_data in MotionDataset.query_range(
                        tables, self.indices, "sample_data", sample["token"],
                        column_name="sample_token"
                    )
                    if MotionDataset.check_sensor(tables, self.indices, sample_data, channel) and
                    key_filter(sample_data)
                ], key=lambda x: x["timestamp"])
                for channel in sensor_channels
            ])
            for scene in tables["scene"]
        ]

        # 构建clips（包含motion frames）
        self.items = self._build_clips_with_motion_frames(
            scene_channel_sample_data, fps_stride_tuples
        )

        # 序列化大表以节省内存
        self.tables = {
            k: (
                dwm.common.SerializedReadonlyList(v)
                if k in self.serialized_table_names else v
            )
            for k, v in tables.items()
        }

        # cache the map data
        if self.hdmap_image_settings is not None:
            self.map_expansion = {}
            self.map_expansion_dict = {}
            for i in tables["log"]:
                to_dict = ["node", "polygon"]
                if i["location"] not in self.map_expansion:
                    name = "expansion/{}.json".format(i["location"])
                    self.map_expansion[i["location"]] = json.loads(
                        fs.cat_file(name).decode())
                    self.map_expansion_dict[i["location"]] = {}
                    for j in to_dict:
                        self.map_expansion_dict[i["location"]][j] = {
                            k["token"]: k
                            for k in self.map_expansion[i["location"]][j]
                        }
        else:
            self.map_expansion = {}
            self.map_expansion_dict = {}

        # 加载image description
        if image_description_settings is not None:
            with open(
                image_description_settings["path"], "r", encoding="utf-8"
            ) as f:
                self.image_descriptions = json.load(f)

            self.image_desc_rs = np.random.RandomState(
                image_description_settings["seed"]
                if "seed" in image_description_settings else None)

            with open(
                image_description_settings["time_list_dict_path"], "r",
                encoding="utf-8"
            ) as f:
                self.time_list_dict = json.load(f)

    def _build_clips_with_motion_frames(self, scene_channel_sample_data, fps_stride_tuples):
        """
        构建包含motion frames的clips

        每个clip包含:
        - ref_length个motion frames（从前面的帧采样）
        - video_length个实际帧
        """
        from dwm.datasets.nuscenes import MotionDataset

        all_items = []

        for scene, channel_sample_data in scene_channel_sample_data:
            scene_token = scene["token"]
            csdl = channel_sample_data

            if not csdl or not csdl[0]:
                continue

            # 遍历不同的fps和stride组合
            for fps, stride in fps_stride_tuples:
                # 枚举所有可能的clip起点
                max_start = max(0, len(csdl[0]) - self.video_length + 1)

                for start in range(0, max_start):
                    # 如果要求从keyframe开始
                    if self.start_on_keyframe:
                        if not csdl[0][start].get("is_key_frame", False):
                            continue

                    # 采样motion frames
                    ref_indices = _sample_motion_frames(
                        list(range(len(csdl[0]))),
                        start,
                        self.ref_length,
                        self.candidate_length
                    )

                    # 构建视频序列
                    video_indices = list(range(start, min(start + self.video_length, len(csdl[0]))))

                    # 确保视频长度正确
                    if len(video_indices) < self.video_length:
                        # 如果不够，重复最后一帧
                        video_indices.extend([video_indices[-1]] * (self.video_length - len(video_indices)))

                    # 合并motion和video索引
                    all_indices = ref_indices + video_indices

                    # 对每个相机构建token序列
                    segment_tokens = []
                    for t_idx in all_indices:
                        frame_tokens = []
                        for channel_idx, channel_data in enumerate(csdl):
                            if t_idx < len(channel_data):
                                frame_tokens.append(channel_data[t_idx]["token"])
                            else:
                                # 越界情况：使用最后一帧
                                frame_tokens.append(channel_data[-1]["token"])
                        segment_tokens.append(frame_tokens)

                    all_items.append({
                        "segment": segment_tokens,
                        "fps": fps,
                        "scene": scene_token,
                        "ref_length": self.ref_length,
                        "video_length": self.video_length,
                        "start_idx": start,
                        "ref_indices": ref_indices,
                        "video_indices": video_indices,
                    })

        return dwm.common.SerializedReadonlyList(all_items)

    def __len__(self):
        return len(self.items)

    def _query(self, table_name: str, key: str, column_name: str = "token"):
        """查询表"""
        from dwm.datasets.nuscenes import MotionDataset
        return MotionDataset.query(self.tables, self.indices, table_name, key, column_name)

    def _check_sensor(self, sample_data: dict, channel=None, modality=None):
        """检查传感器"""
        from dwm.datasets.nuscenes import MotionDataset
        return MotionDataset.check_sensor(self.tables, self.indices, sample_data, channel, modality)

    def __getitem__(self, index: int):
        """
        返回一个clip的数据

        返回的result包含:
        - images: [T, V, H, W, 3] 多视图图像
        - motion_images: [M, V, H, W, 3] motion frames (M=ref_length)
        - video_images: [V_len, V, H, W, 3] video frames
        - relative_poses: [T, 4, 4] 相对位姿
        - camera_intrinsics: [T, V, 3, 3]
        - camera2ego: [T, V, 4, 4]
        - (可选) ref_images, bev_images, layout_canvas等
        """
        from dwm.datasets.nuscenes import MotionDataset

        item = self.items[index]
        scene = self._query("scene", item["scene"])

        # 获取segment中的所有sample_data
        segment = [
            [self._query("sample_data", token) for token in frame_tokens]
            for frame_tokens in item["segment"]
        ]

        total_frames = len(segment)
        ref_length = item["ref_length"]
        video_length = item["video_length"]

        # 分离motion frames和video frames
        motion_segment = segment[:ref_length]
        video_segment = segment[ref_length:]

        result = {
            "fps": torch.tensor(item["fps"], dtype=torch.float32),
            "ref_length": ref_length,
            "video_length": video_length,
            "motion_indices": item["ref_indices"],
            "video_indices": item["video_indices"],
        }

        if self.enable_sample_data:
            result["sample_data"] = segment
            result["scene"] = scene

        # 计算时间戳（相对于第一帧）
        first_timestamp = motion_segment[0][0]["timestamp"]
        result["pts"] = torch.tensor([
            [
                (sd["timestamp"] - first_timestamp + 500) // 1000
                for sd in frame
            ]
            for frame in segment
        ], dtype=torch.float32)

        # ========== 加载图像 ==========
        images = []
        for frame in segment:
            frame_images = []
            for sd in frame:
                if self._check_sensor(sd, modality="camera"):
                    with self.fs.open(sd["filename"]) as f:
                        img = Image.open(f)
                        img.load()
                    frame_images.append(img)

            # 确保有6个视图
            while len(frame_images) < 6:
                if frame_images:
                    frame_images.append(frame_images[-1])
                else:
                    # 完全空的帧，用黑色图填充
                    H, W = self.layout_canvas_size
                    frame_images.append(Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)))

            if frame_images:
                images.append(frame_images)

        if images:
            result["images"] = images
            # motion_frames和video_indices信息已经保存在item中
            # DreamForge的model会在pipeline层面处理motion frames逻辑

        # ========== 计算相对位姿（DreamForge特性）==========
        first_info = segment[0][0]  # 以第一个motion frame为参考
        relative_poses = []
        for frame in segment:
            # 使用lidar信息计算相对位姿
            lidar_sd = next((sd for sd in frame if self._check_sensor(sd, modality="lidar")), frame[0])
            relative_pose = _compute_relative_pose(first_info, lidar_sd)
            relative_poses.append(torch.from_numpy(relative_pose))

        result["relative_poses"] = torch.stack(relative_poses, dim=0)  # (T, 4, 4)

        # ========== Camera Transforms ==========
        if self.enable_camera_transforms and images:
            camera_intrinsics = []
            camera2ego = []
            lidar2image = []

            for frame in segment:
                frame_intrinsics = []
                frame_c2e = []
                frame_l2i = []

                for sd in frame:
                    if not self._check_sensor(sd, modality="camera"):
                        continue

                    calib = self._query("calibrated_sensor", sd["calibrated_sensor_token"])

                    # 内参
                    K = np.array(calib["camera_intrinsic"], dtype=np.float32)
                    frame_intrinsics.append(torch.from_numpy(K))

                    # camera2ego变换
                    c2e = dwm.datasets.common.get_transform(
                        calib["rotation"], calib["translation"], "np"
                    ).astype(np.float32)
                    frame_c2e.append(torch.from_numpy(c2e))

                    # lidar2image（用于投影）
                    lidar_sd = next((s for s in frame if self._check_sensor(s, modality="lidar")), None)
                    if lidar_sd:
                        lidar_calib = self._query("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
                        l2e = dwm.datasets.common.get_transform(
                            lidar_calib["rotation"], lidar_calib["translation"], "np"
                        )
                        e2c = np.linalg.inv(c2e)
                        K_homo = np.eye(4, dtype=np.float32)
                        K_homo[:3, :3] = K
                        l2i = K_homo @ e2c @ l2e
                        frame_l2i.append(torch.from_numpy(l2i.astype(np.float32)))

                # 确保每个帧有6个相机数据
                while len(frame_intrinsics) < 6:
                    if frame_intrinsics:
                        frame_intrinsics.append(frame_intrinsics[-1])
                        frame_c2e.append(frame_c2e[-1])
                    else:
                        # 默认内参和变换
                        frame_intrinsics.append(torch.eye(3, dtype=torch.float32))
                        frame_c2e.append(torch.eye(4, dtype=torch.float32))

                while len(frame_l2i) < 6:
                    if frame_l2i:
                        frame_l2i.append(frame_l2i[-1])
                    else:
                        frame_l2i.append(torch.eye(4, dtype=torch.float32))

                camera_intrinsics.append(torch.stack(frame_intrinsics))
                camera2ego.append(torch.stack(frame_c2e))
                lidar2image.append(torch.stack(frame_l2i))

            if camera_intrinsics:
                result["camera_intrinsics"] = torch.stack(camera_intrinsics)  # (T, 6, 3, 3)
                camera2ego_stack = torch.stack(camera2ego)  # (T, 6, 4, 4)
                result["camera2ego"] = camera2ego_stack
                result["camera_transforms"] = camera2ego_stack  # 别名，兼容cross_attention pipeline
                result["lidar2image"] = torch.stack(lidar2image)  # (T, 6, 4, 4)

        # ========== Ego Transforms ==========
        if self.enable_ego_transforms:
            ego_transforms = []
            for frame in segment:
                frame_et = []
                for sd in frame:
                    rot = sd.get("rotation", sd.get("ego_rotation", [0, 0, 0, 1]))
                    trans = sd.get("translation", sd.get("ego_translation", [0, 0, 0]))
                    et = dwm.datasets.common.get_transform(rot, trans, "pt")
                    frame_et.append(et)

                # 确保有6个视图
                while len(frame_et) < 6:
                    if frame_et:
                        frame_et.append(frame_et[-1])
                    else:
                        frame_et.append(torch.eye(4, dtype=torch.float32))

                ego_transforms.append(torch.stack(frame_et))

            if ego_transforms:
                result["ego_transforms"] = torch.stack(ego_transforms)

        # ========== Image Size (用于 explicit_view_modeling) ==========
        if self.enable_camera_transforms:
            # 提取每个相机图像的尺寸 [height, width]
            image_sizes = []
            for frame in segment:
                frame_sizes = []
                for sd in frame:
                    if self._check_sensor(sd, modality="camera"):
                        frame_sizes.append(torch.tensor([sd.get("height", 900), sd.get("width", 1600)], dtype=torch.long))

                # 确保有6个视图
                while len(frame_sizes) < 6:
                    if frame_sizes:
                        frame_sizes.append(frame_sizes[-1])
                    else:
                        frame_sizes.append(torch.tensor(self.layout_canvas_size, dtype=torch.long))

                image_sizes.append(torch.stack(frame_sizes))

            if image_sizes:
                result["image_size"] = torch.stack(image_sizes)  # (T, V, 2)

        # ========== Layout Canvas (DreamForge特性) ==========
        if self.generate_layout:
            layout_list = generate_layout_canvas(
                segment,
                map_classes=len(self.map_classes),
                object_classes=len(self.object_classes),
                image_size=self.layout_canvas_size
            )
            result["layout_canvas"] = layout_list  # List[T, V, 13, H, W]

        # ========== HDMap & 3DBox Images (如果有配置) ==========
        if self._3dbox_image_settings is not None:
            # 从缓存加载或生成3D box图像
            camera_sdl_per_t = [
                [j for j in sdl if self._check_sensor(j, modality="camera")]
                for sdl in segment
            ]

            all_hit = True
            cached = []
            for sdl in camera_sdl_per_t:
                row = []
                for sd in sdl:
                    p = _png_path(self.cache_root, "3dbox_images", sd["token"])
                    if os.path.isfile(p):
                        img = _try_open_png(p)
                        if img is not None:
                            row.append(img)
                        else:
                            all_hit = False
                            row.append(None)
                    else:
                        all_hit = False
                        row.append(None)
                cached.append(row)

            if all_hit:
                result["3dbox_images"] = cached
            else:
                # 从settings生成3D box图像
                result["3dbox_images"] = [
                    [
                        MotionDataset.get_3dbox_image(
                            self.tables, self.indices, sd, self._3dbox_image_settings
                        )
                        for sd in sdl
                    ]
                    for sdl in camera_sdl_per_t
                ]

        if self.hdmap_image_settings is not None:
            camera_sdl_per_t = [
                [j for j in sdl if self._check_sensor(j, modality="camera")]
                for sdl in segment
            ]

            all_hit = True
            cached = []
            for sdl in camera_sdl_per_t:
                row = []
                for sd in sdl:
                    p = _png_path(self.cache_root, "hdmap_images", sd["token"])
                    if os.path.isfile(p):
                        img = _try_open_png(p)
                        if img is not None:
                            row.append(img)
                        else:
                            all_hit = False
                            row.append(None)
                    else:
                        all_hit = False
                        row.append(None)
                cached.append(row)

            if all_hit:
                result["hdmap_images"] = cached
            else:
                result["hdmap_images"] = [
                    [
                        MotionDataset.get_hdmap_image(
                            self.map_expansion, self.map_expansion_dict,
                            self.tables, self.indices, sd, self.hdmap_image_settings
                        )
                        for sd in sdl
                    ]
                    for sdl in camera_sdl_per_t
                ]

        # ========== Image Description ==========
        if self.image_description_settings is not None:
            from dwm.datasets.nuscenes import MotionDataset
            camera_sdl_per_t = [
                [j for j in sdl if self._check_sensor(j, modality="camera")]
                for sdl in segment
            ]
            image_captions = [
                dwm.datasets.common.align_image_description_crossview([
                    MotionDataset.get_image_description(
                        self.tables, self.indices, self.image_descriptions,
                        self.time_list_dict, item["scene"], sd)
                    for sd in sdl
                ], self.image_description_settings)
                for sdl in camera_sdl_per_t
            ]
            result["image_description"] = [
                [
                    dwm.datasets.common.make_image_description_string(
                        j, self.image_description_settings, self.image_desc_rs)
                    for j in i
                ]
                for i in image_captions
            ]

        # ========== Stub数据 ==========
        if self.stub_key_data_dict:
            dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)

        return result


# ============================================================================
# Collate函数
# ============================================================================

def collate_fn_dreamforge(batch, drop_keys=None):
    """
    DreamForge版本的collate函数

    处理:
    1. 堆叠tensor数据
    2. 保持list数据（如images）
    3. 对齐不同长度的数据
    """
    if not batch:
        return None

    drop_keys = drop_keys or []

    result = {}

    # 简单的tensor堆叠
    tensor_keys = ["fps", "pts", "relative_poses", "camera_intrinsics",
                   "camera2ego", "camera_transforms", "lidar2image", "ego_transforms",
                   "image_size", "ref_length", "video_length"]

    for key in tensor_keys:
        if key in batch[0] and key not in drop_keys:
            # 检查是否所有batch都有这个key
            if all(key in item and item[key] is not None for item in batch):
                result[key] = torch.stack([item[key] for item in batch])

    # 保持list结构的数据
    list_keys = ["images", "motion_images", "video_images", "sample_data", "scene", "layout_canvas"]

    for key in list_keys:
        if key in batch[0] and key not in drop_keys:
            result[key] = [item[key] for item in batch if key in item]

    # ref_images 和 bev_images 是列表的列表
    list_of_list_keys = ["ref_images", "bev_images", "3dbox_images", "hdmap_images"]

    for key in list_of_list_keys:
        if key in batch[0] and key not in drop_keys:
            # 保持为列表的列表结构
            result[key] = [item[key] for item in batch if key in item]

    return result


__all__ = ["MotionDatasetDreamForge", "collate_fn_dreamforge", "obtain_next2top"]
