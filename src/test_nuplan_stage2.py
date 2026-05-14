"""
NuPlan仿真 - 阶段2：集成Diffusion生成模型
目标：在阶段1基础上，恢复图像生成功能，验证闭环生成
"""
import os
import numpy as np
from typing import List
import logging
import pickle
import torch
import time
import ray
import torchvision
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from pathlib import Path
import yaml
import cv2

from pyquaternion import Quaternion

# ========== 添加这段代码 ==========
# 修复nuscenes matplotlib样式兼容性问题
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle



# 检查是否存在seaborn-v0_8-whitegrid样式，如果不存在则使用替代样式
if 'seaborn-v0_8-whitegrid' not in plt.style.available:
    # 使用seaborn-whitegrid或其他可用样式作为替代
    available_seaborn_styles = [s for s in plt.style.available if 'seaborn' in s and 'whitegrid' in s]
    if available_seaborn_styles:
        # 预先设置样式，避免nuscenes设置时出错
        plt.style.use(available_seaborn_styles[0])
    else:
        # 使用默认的whitegrid样式
        plt.style.use('default')
    
    # Monkey patch nuscenes的样式设置
    original_use = plt.style.use
    def patched_use(style):
        if style == 'seaborn-v0_8-whitegrid':
            # 使用可用的替代样式
            if available_seaborn_styles:
                style = available_seaborn_styles[0]
            else:
                style = 'default'
        return original_use(style)
    plt.style.use = patched_use
# ========== 结束添加 ==========

# 加载路径配置
config_path = Path(__file__).parent / "config_paths.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"配置文件不存在: {config_path}")

with open(config_path, 'r', encoding='utf-8') as f:
    path_config = yaml.safe_load(f)

# 设置环境变量
os.environ["NUPLAN_DATA_ROOT"] = path_config['nuplan_data_root']
os.environ["NUPLAN_MAPS_ROOT"] = path_config['nuplan_maps_root']
os.environ["NUPLAN_DB_FILES"] = path_config['nuplan_db_files']
os.environ["NUPLAN_MAP_VERSION"] = path_config['nuplan_map_version']
os.environ["NUPLAN_EXP_ROOT"] = path_config['nuplan_exp_root']
os.environ['BLOB_PATH'] = path_config['blob_path']
os.environ["NUPLAN_DATA_STORE"] = path_config.get('nuplan_data_store', '')

print("=" * 80)
print("NuPlan仿真 - 阶段2：集成Diffusion生成模型")
print("=" * 80)

from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.observation_builder import build_observations
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.planner_builder import build_planners
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from hydra.utils import instantiate

from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInput
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.observation.observation_type import CameraChannel
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.database.nuplan_db_orm.camera import Camera
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

# Magicdrive imports
from magicdrive.misc.test_utils import build_pipe
from magicdrive.pipeline.pipeline_bev_controlnet import BEVStableDiffusionPipelineOutput
from PIL import Image as PILImage

# 本地imports
from common import (
    rotate_round_z_axis, 
    resize_img, 
    get_transmat_for_lidarpc_token_from_db,
    obtain_sensor2top,
    get_aug_mat,
    global_trajectory_to_states
)
from proj_ctrls_utils import (
    get_projected_map,
    get_projected_bboxes,
    get_ds_map,
)

import matplotlib.pyplot as plt
from hydra import initialize, compose
from omegaconf import OmegaConf

# 初始化Hydra
with initialize(version_base=None, config_path="./configs"):
    cfg = compose(config_name="simulation_config.yaml", overrides=[
        "+simulation=closed_loop_reactive_agents",
        "planner=simple_planner",
        "scenario_builder=my_nuplan_mini_debug",
        "scenario_filter=test_random14",
        "worker.threads_per_node=4",
        "experiment_uid=test_random14/simple_planner_stage2",
        "verbose=true",
    ])

# 初始化Ray
ray_temp_dir = Path(path_config['ray_temp_dir'])
ray_temp_dir.mkdir(parents=True, exist_ok=True)
ray.init(_temp_dir=str(ray_temp_dir))

# 构建scenarios
worker = build_worker(cfg)
scenario_builder = build_scenario_builder(cfg=cfg)
scenario_filter = build_scenario_filter(cfg=cfg.scenario_filter)
scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

print(f"找到 {len(scenarios)} 个场景")

# 颜色配置
colors = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
    [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 0.5, 0],
]

post_trans_cat = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

def proj_map_vis(proj_map):
    """可视化投影地图"""
    proj_map = proj_map.cpu().numpy()
    H, W = proj_map.shape[1:]
    box_ctrls = np.zeros((H, W, 3), dtype=np.uint8)
    
    for j in range(proj_map.shape[0]):
        mask = proj_map[j] == 1
        for c in range(3):  
            box_ctrls[..., c] += mask.astype(np.uint8) * colors[j][c]
    
    box_ctrls = np.transpose(box_ctrls, (2,0,1))
    return box_ctrls

def process_depth_map(depth):
    """处理深度图"""
    depth = depth.copy()
    mask_invalid = depth == -300
    valid_mask = ~mask_invalid

    if np.any(valid_mask):
        valid_values = depth[valid_mask]
        min_val = valid_values.min()
        max_val = valid_values.max()
        if max_val > min_val:
            depth[valid_mask] = (valid_values - min_val) / (max_val - min_val)
        else:
            depth[valid_mask] = 0.0

    depth[mask_invalid] = -1.0
    return depth

def visualize_depth_map(depth_map, colormap=cv2.COLORMAP_INFERNO):
    """可视化深度图"""
    depth_map = depth_map.cpu().numpy()
    min_val, max_val = np.min(depth_map), np.max(depth_map)
    normalized_depth = (255 * (depth_map - min_val) / (max_val - min_val)).astype(np.uint8)
    depth_colored = cv2.applyColorMap(normalized_depth, colormap)
    return depth_colored

def _get_global_trajectory(local_trajectory: np.ndarray, ego_state):
    """局部轨迹转全局轨迹"""
    origin = ego_state.rear_axle.array
    angle = ego_state.rear_axle.heading

    global_position = (
        rotate_round_z_axis(np.ascontiguousarray(local_trajectory[..., :2]), -angle)
        + origin
    )
    global_vec = global_position[1:] - global_position[:-1]
    global_heading = np.arctan2(global_vec[..., 1], global_vec[..., 0])
    global_heading = np.concatenate([np.array([angle]), global_heading])

    global_trajectory = np.concatenate(
        [global_position, global_heading[..., None]], axis=1
    )
    return global_trajectory



def get_traj(ego_state, ego_state_buffer, trajectory_type="straight"):
    """
    生成预定义轨迹
    
    Args:
        ego_state: 当前ego状态
        ego_state_buffer: ego状态历史缓冲
        trajectory_type: 轨迹类型，可选 "straight" (直行) 或 "cosine" (余弦横向偏移)
    """
    num_points = 60
    length = 30.0
    x = np.linspace(0, length, num_points)
    
    if trajectory_type == "straight":
        # 直行轨迹：横向偏移为0
        y = np.zeros_like(x)
    elif trajectory_type == "cosine":
        # 余弦轨迹：横向偏移
        lateral_offset = -1.5
        y = lateral_offset * 0.5 * (1 - np.cos(np.pi * x / length))
    else:
        raise ValueError(f"不支持的轨迹类型: {trajectory_type}，请选择 'straight' 或 'cosine'")
    
    local_traj = np.stack([x, y], axis=1)

    global_traj = _get_global_trajectory(local_traj, ego_state)

    traj = InterpolatedTrajectory(
        trajectory=global_trajectory_to_states(
            global_trajectory=global_traj,
            ego_history=ego_state_buffer,
            future_horizon=len(global_traj) * 0.1,
            step_interval=0.1,
        )
    )
    return traj


logger = logging.getLogger(__name__)

def img_wconcat_save(tmp, gen_imgs_list, post_trans, auto_id, scene_id, save_dir):
    """保存拼接的图像（8视角+控制信号）以及单独的中间图像"""
    camera_names = ['CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_B0', 'CAM_R2', 'CAM_R1', 'CAM_R0']
    
    for bi, template in enumerate(tmp):
        for gen_id, gen_imgs in enumerate(gen_imgs_list):
            # 准备拼接图像的各个组件
            clr_map = torch.cat([tmp['clr_map'][i] for i in range(8)], dim=2).to('cpu') 
            depth_map = torch.from_numpy(visualize_depth_map(
                torch.cat([tmp['depth_map'][i] for i in range(8)], dim=1).to('cpu')
            )).permute(2, 0, 1)
            gt_proj = torch.from_numpy(proj_map_vis(
                torch.cat([tmp['proj_map'][i] for i in range(8)], dim=2)
            ))
            sem_map = (torch.cat([tmp['sem_map'][i] for i in range(8)], dim=2))
            
            # 创建子目录结构
            step_dir = Path(save_dir) / f"step_{auto_id:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存单独的生成图像（8个视角）
            individual_dir = step_dir / "generated_views"
            individual_dir.mkdir(exist_ok=True)
            
            img_list = []
            for idx in range(8):
                img = post_trans(gen_imgs[idx])
                img_list.append(img)
                
                # 保存单独视角的生成图像
                individual_img = to_pil_image(img)
                individual_save_path = individual_dir / f"{camera_names[idx]}.jpg"
                individual_img.save(individual_save_path)
            
            # 保存控制信号
            controls_dir = step_dir / "controls"
            controls_dir.mkdir(exist_ok=True)
            
            # 保存深度图
            depth_img = to_pil_image(depth_map)
            depth_img.save(controls_dir / "depth_map.jpg")
            
            # 保存颜色地图
            clr_img = to_pil_image(clr_map)
            clr_img.save(controls_dir / "clr_map.jpg")
            
            # 保存投影地图
            proj_img = to_pil_image(gt_proj)
            proj_img.save(controls_dir / "proj_map.jpg")
            
            # 保存语义地图
            sem_img = to_pil_image(sem_map)
            sem_img.save(controls_dir / "sem_map.jpg")
            
            # 保存单独的控制信号（分视角）
            controls_per_view_dir = step_dir / "controls_per_view"
            controls_per_view_dir.mkdir(exist_ok=True)
            
            for idx in range(8):
                # 深度图（单视角）
                depth_single = visualize_depth_map(tmp['depth_map'][idx].to('cpu'))
                depth_single_img = PILImage.fromarray(depth_single)
                depth_single_img.save(controls_per_view_dir / f"{camera_names[idx]}_depth.jpg")
                
                # 颜色地图（单视角）
                clr_single = to_pil_image(tmp['clr_map'][idx].to('cpu'))
                clr_single.save(controls_per_view_dir / f"{camera_names[idx]}_clr.jpg")
                
                # 投影地图（单视角）
                proj_single = proj_map_vis(tmp['proj_map'][idx])
                #proj_single_img = to_pil_image(proj_single)
                proj_single_img = to_pil_image(torch.from_numpy(proj_single))
                proj_single_img.save(controls_per_view_dir / f"{camera_names[idx]}_proj.jpg")
                
                # 语义地图（单视角）
                sem_single = to_pil_image(tmp['sem_map'][idx])
                sem_single.save(controls_per_view_dir / f"{camera_names[idx]}_sem.jpg")

            # 保存拼接的大图（所有视角+控制信号）
            img_cat = torch.cat(img_list, dim=2) 
            img_cat = torch.cat([depth_map, clr_map, img_cat, gt_proj, sem_map], dim=1)
            img_cat = to_pil_image(img_cat)

            concat_save_path = step_dir / f"8view_concat_{scene_id}_{auto_id}.jpg"
            img_cat.save(concat_save_path)
            
            print(f"  ✓ 保存步骤 {auto_id} 的所有图像:")
            print(f"    - 拼接图: {concat_save_path}")
            print(f"    - 单视角: {individual_dir}/ (8张)")
            print(f"    - 控制信号: {controls_dir}/ (4张)")
            print(f"    - 分视角控制: {controls_per_view_dir}/ (32张)")


class DiffusionSimulationRunner(SimulationRunner):
    """
    阶段2：集成Diffusion生成的仿真运行器
    """
    def __init__(
        self, 
        cfg,
        simulation: Simulation, 
        planner: AbstractPlanner,
        kernel_size=(1, 1),
        sigma=1.0,
        patch_radius=100,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        ref_mean=[0.485, 0.456, 0.406],
        ref_std=[0.229, 0.224, 0.225],
        im_size=(1080, 1920),
        diffusion_size=(224, 400),
        resize_lim=[1/4.8, 1/4.8],
    ):
        super().__init__(simulation, planner)
        self.cfg = cfg
        self.camera_channels = [
            CameraChannel.CAM_F0, CameraChannel.CAM_L0, CameraChannel.CAM_L1, CameraChannel.CAM_L2,
            CameraChannel.CAM_B0, CameraChannel.CAM_R2, CameraChannel.CAM_R1, CameraChannel.CAM_R0,
        ]
        self.diffusion_size = diffusion_size
        self.resize_lim = resize_lim
        self.patch_radius = patch_radius
        self.im_size = im_size

        self.prev_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ])
        self.ref_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=ref_mean, std=ref_std),
        ])
        self.post_trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (1075, 1920),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.Pad([0,5,0,0]),
        ])

        self.polygon_layer_names = [
            SemanticMapLayer.LANE, SemanticMapLayer.CROSSWALK, SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE, SemanticMapLayer.WALKWAYS, SemanticMapLayer.CARPARK_AREA,
        ]
        self.line_layer_names = [
            SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR,
        ]
        self.obj_classes = [
            TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE,
            TrackedObjectType.TRAFFIC_CONE, TrackedObjectType.BARRIER, TrackedObjectType.CZONE_SIGN
        ]
        
        # 加载场景点云和actor模型
        clr_scene_path = Path(path_config['clr_scene_file'])
        if clr_scene_path.exists():
            self.clr_scene = np.load(str(clr_scene_path), allow_pickle=False)
            print(f"加载场景点云: {clr_scene_path}")
        else:
            print(f"警告: 场景点云文件不存在: {clr_scene_path}")
            self.clr_scene = None
            
        actor_root = Path(path_config['actor_root'])
        self.actor_root = str(actor_root)
        if actor_root.exists():
            self.static_actors = {
                'sedan': pickle.load(open(actor_root / 'sedan.pkl', "rb")),
                'suv': pickle.load(open(actor_root / 'suv.pkl', "rb")),
                'pickup': pickle.load(open(actor_root / 'pickup.pkl', "rb")),
                'bike': pickle.load(open(actor_root / 'bike.pkl', "rb")),
                'ped': pickle.load(open(actor_root / 'ped.pkl', "rb")),
            }
            print(f"加载Actor模型: {actor_root}")
        else:
            print(f"警告: Actor目录不存在: {actor_root}")
            self.static_actors = {}
        
    def _initialize(self) -> None:
        """初始化"""
        print("\n[初始化] 开始初始化仿真（阶段2）...")
        
        self._simulation.callback.on_initialization_start(self._simulation.setup, self.planner)
        self.planner.initialize(self._simulation.initialize())
        self._simulation.callback.on_initialization_end(self._simulation.setup, self.planner)

        # 加载Diffusion pipeline
        print("[初始化] 加载Diffusion pipeline...")
        try:
            self.pipeline, self.weight_dtype = build_pipe(self.cfg, device="cuda")
            print("[初始化] Pipeline加载成功")
        except Exception as e:
            print(f"[错误] Pipeline加载失败: {e}")
            raise

        # 加载数据库和地图
        self.db_wrapper = NuPlanDBWrapper(
            self.scenario._data_root, 
            self.scenario._map_root, 
            self.scenario._log_file_load_path, 
            self.scenario._map_version
        )
        db_name = Path(self.scenario._log_file_load_path).stem
        self.db_record = self.db_wrapper.get_log_db(db_name)
        map_factory = NuPlanMapFactory(get_maps_db(
            map_root=self.scenario._map_root, 
            map_version=self.scenario._map_version
        ))
        self.numap = map_factory.build_map_from_name(self.scenario._map_name)

        location = self.db_record.log.location
        self.caption = f"A driving scene image at {location}"

        self.prepare_ego_coords()
        print("[初始化] 完成")
        
    def prepare_ego_coords(self):
        """准备ego坐标"""
        ego_coords = []
        for it in range(self.simulation.scenario.get_number_of_iterations()):
            ego_state = self.simulation.scenario.get_ego_state_at_iteration(it)
            ego_coords.append(ego_state.rear_axle.point.array)
        self.scene_ego_coords = np.stack(ego_coords)

    def get_ref_index(self, ego_coord, velo):
        """获取参考帧索引"""
        vec = self.scene_ego_coords - ego_coord
        score = np.sum(vec * velo, axis=-1)

        positive_indices = np.where(score >= 0)[0]
        negative_indices = np.where(score < 0)[0]

        if positive_indices.size > 0:
            min_positive_index = positive_indices[np.argmin(score[positive_indices])]
        else:
            min_positive_index = negative_indices[np.argmax(score[negative_indices])]

        if negative_indices.size > 0:
            max_negative_index = negative_indices[np.argmax(score[negative_indices])]
        else:
            max_negative_index = positive_indices[np.argmin(score[positive_indices])]

        return min_positive_index, max_negative_index

    def get_ref_sensors(self, idx, num, step):
        """获取参考传感器数据"""
        sensors = self.simulation.scenario.get_sensors_at_iteration(idx, self.camera_channels)
        imgs = [sensors.images[ch].as_pil for ch in self.camera_channels]
        return torch.stack([
            self.ref_transform(resize_img(img, self.resize_lim[0], self.diffusion_size))
            for img in imgs
        ])

    def get_ego2global(self, ego_state, initial_ego2global):
        """获取ego到全局的变换矩阵"""
        trans_z = initial_ego2global[2,3]
        ego2global = ego_state.rear_axle.as_matrix_3d()
        ego2global[2,3] = trans_z
        return ego2global

    def get_sensor_metas(self, ego2global):
        """获取传感器元数据"""
        info = {"cam":{}}
        for cam in self.camera_channels:
            cam_db: Camera = self.db_record.camera.select_one(channel = cam.value)
            cam_info = obtain_sensor2top(cam_db, self.db_record.lidar[0], ego2global)
            info["cam"].update({cam: cam_info})

        h, w = self.im_size

        ret = {}
        ret["lidar2camera"] = []
        ret["lidar2image"] = []
        ret["camera_intrinsics"] = []
        ret["camera2lidar"] = []
        ret["sensor2ego_translation"] = []
        ret["sensor2ego_rotation"] = []
        ret["distortion"] = []
        
        for _, camera_info in info["cam"].items():
            ret["distortion"].append(np.array(camera_info["distortion"]))
            
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (camera_info["sensor2lidar_translation"] @ lidar2camera_r.T)
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            ret["lidar2camera"].append(lidar2camera_rt.T)

            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            distortion = ret["distortion"][-1]
            cam_intrin = camera_intrinsics[:3, :3]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_intrin, distortion, (w, h), 1, (w, h))
            new_cam_mat_pad = np.pad(new_camera_matrix, ((0, 1), (0, 1)), mode='constant')
            new_cam_mat_pad[3, 3] = 1
            ret["camera_intrinsics"].append(new_cam_mat_pad)

            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            ret["lidar2image"].append(lidar2image)
            
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            ret["camera2lidar"].append(camera2lidar)

            ret["sensor2ego_translation"].append(camera_info["sensor2ego_translation"])
            ret["sensor2ego_rotation"].append(camera_info["sensor2ego_rotation"])
        return ret

    def get_object_bboxes(self, observation, ego_state):
        """获取目标边界框"""
        tracked_objs: TrackedObjects = observation.tracked_objects
        ego2global = ego_state.rear_axle.as_matrix()
        gt_bboxes = []
        gt_labels = []
        gt_track = []
        gt_actorbbox = []
        
        for obj_cls in self.obj_classes:
            obj_list = tracked_objs.get_tracked_objects_of_type(obj_cls)
            if len(obj_list) == 0:
                continue
            box_list = [obj.box for obj in obj_list]
            track_token = [obj.track_token for obj in obj_list]
            locs_2d = np.array([b.center.point.array for b in box_list]).reshape(-1, 2)
            locs_2d = locs_2d - ego2global[:2, 2].reshape(-1, 2)
            locs_2d = locs_2d @ np.linalg.inv(ego2global[:2, :2]).T
            locs = np.concatenate([locs_2d, np.zeros_like(locs_2d)[..., :1]], axis=-1)
            dims = np.array([[b.width, b.length, b.height] for b in box_list]).reshape(-1, 3)
            rots = np.array([b.center.heading - ego_state.rear_axle.heading for b in box_list]).reshape(-1, 1)
            boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            dims[:,[0,1]] = dims[:,[1,0]]
            actor_boxes = np.concatenate([locs, dims, rots], axis=1)

            labels = int(obj_cls) * np.ones(len(boxes))
            gt_bboxes.append(boxes)
            gt_labels.append(labels)
            gt_track.append(track_token)
            gt_actorbbox.append(actor_boxes)
            
        if len(gt_bboxes) != 0:
            gt_bboxes = np.concatenate(gt_bboxes, axis=0)
            gt_labels = np.concatenate(gt_labels, axis=0)
            gt_bboxes = LiDARInstance3DBoxes(
                gt_bboxes, box_dim=gt_bboxes.shape[-1], origin=(0.5, 0.5, 0)
            )
            gt_track = np.concatenate(gt_track, axis=0)
            gt_actorbbox = np.concatenate(gt_actorbbox, axis=0)

        return gt_bboxes, gt_labels, gt_track, gt_actorbbox

    def get_diffused_img(self, planner_input: PlannerInput, prev_sensor_data: List[PILImage.Image], step):
        """生成扩散图像"""
        print(f"  [生成] 开始生成第{step}步图像...")
        
        assert len(planner_input.history.ego_state_buffer) > 1
        ego_state = planner_input.history.ego_state_buffer[-1]
        ego_pose = ego_state.center
        
        initial_ego2global = get_transmat_for_lidarpc_token_from_db(
            self.simulation.scenario._log_file, self.simulation.scenario._initial_lidar_token
        )
        ego2global = self.get_ego2global(ego_state, initial_ego2global)

        velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d.array
        acceleration = ego_state.dynamic_car_state.rear_axle_acceleration_2d.array
        angular_v = ego_state.dynamic_car_state.angular_velocity
        prev_ego_feats = torch.tensor([velocity[0], velocity[1], acceleration[0], acceleration[1], angular_v])

        # prev_img
        prev_img = [
            self.prev_transform(resize_img(img, self.resize_lim[0], self.diffusion_size))
            for img in prev_sensor_data
        ]
        prev_img = torch.stack(prev_img)
        prev_ego_state = planner_input.history.ego_state_buffer[-2]
        prev_ego2global = self.get_ego2global(prev_ego_state, initial_ego2global)
        prev_trans_mat = torch.from_numpy(np.linalg.inv(ego2global) @ (prev_ego2global))

        # ref_img
        ego_coord = ego_state.rear_axle.point.array
        velo = rotate_round_z_axis(velocity, -ego_state.rear_axle.heading)
        velo = velo / np.linalg.norm(velo)
        
        ref_indices = self.get_ref_index(ego_coord, velo)
        ref_img = []
        ref_trans_mats = []

        for i, index in enumerate(ref_indices):
            ref_img.append(self.get_ref_sensors(index, i, step))
            ref_ego2global = self.get_ego2global(
                self.simulation.scenario.get_ego_state_at_iteration(index), initial_ego2global
            )
            ref_trans_mats.append(torch.from_numpy(np.linalg.inv(ego2global) @ (ref_ego2global)))
            
        ref_front_img, ref_rear_img = ref_img
        ref_front_trans_mat, ref_rear_trans_mat = ref_trans_mats
        ref_front_img = ref_front_img[:,:,:112,:]
        ref_rear_img = ref_rear_img[:,:,:112,:]

        # get projected controls
        sensor_metas = self.get_sensor_metas(ego2global)
        sensor2ego_t_list = sensor_metas["sensor2ego_translation"]
        sensor2ego_r_list = sensor_metas["sensor2ego_rotation"]
        cam_intrinsics_list = sensor_metas["camera_intrinsics"]
        lidar2image_list = sensor_metas["lidar2image"]
        
        map_ctrls = get_projected_map(
            self.numap, self.patch_radius, 8,
            (self.im_size[1], self.im_size[0]), self.diffusion_size,
            ego2global, sensor2ego_t_list, sensor2ego_r_list, cam_intrinsics_list,
            self.polygon_layer_names, self.line_layer_names,
        )

        cur_observation = planner_input.history.observation_buffer[-1]
        gt_bboxes, gt_labels, gt_track, gt_actorbbox = self.get_object_bboxes(cur_observation, ego_state)
        print('gt_bboxes', gt_bboxes)
        print('gt_labels', gt_labels)
        print('gt_track', gt_track)
        print('gt_actorbbox', gt_actorbbox)
        obj_ctrls, _ = get_projected_bboxes(
            gt_bboxes, gt_labels, 8,
            (self.im_size[1], self.im_size[0]), self.diffusion_size,
            lidar2image_list, self.obj_classes, TrackedObjectType.VEHICLE,
        )
        #print('obj_ctrls', obj_ctrls)
      
        if self.clr_scene is not None and self.static_actors:
            sem_map, clr_map, depth_map = get_ds_map(
                ego_pose, ego2global, gt_track, gt_actorbbox, gt_labels, 
                self.clr_scene, self.actor_root, lidar2image_list, self.static_actors
            )
        else:
            print("  [警告] 跳过场景渲染（缺少点云或actor）")
            sem_map = np.zeros((8, 3, 224, 400), dtype=np.uint8)
            clr_map = np.zeros((8, 1, 224, 400, 3), dtype=np.uint8)
            depth_map = np.full((8, 1, 224, 400), -300, dtype=np.float32)

        proj_conds = torch.from_numpy(np.concatenate([
            obj_ctrls, map_ctrls, process_depth_map(depth_map), sem_map
        ] + [clr_map.squeeze(1).transpose(0, 3, 1, 2) / 255], axis=1)).float()

        camera_param = torch.cat([
            torch.from_numpy(np.stack(sensor_metas["camera_intrinsics"])[:, :3, :3]),
            torch.from_numpy(np.stack(sensor_metas["camera2lidar"])[:, :3]),
        ], dim=-1)
        
        sensor_metas["img_aug_matrix"] = torch.stack([get_aug_mat(prev_sensor_data[0], self.resize_lim[0], self.diffusion_size)] * 8)
        to_weight_type = lambda x: x.to(self.weight_dtype).unsqueeze(0)

        controlnet_args = [camera_param, ref_front_trans_mat, ref_rear_trans_mat, prev_trans_mat, proj_conds,
            ref_front_img, ref_rear_img, prev_ego_feats, prev_img]

        camera_param, ref_front_trans_mat, ref_rear_trans_mat, prev_trans_mat, proj_conds, \
            ref_front_img, ref_rear_img, prev_ego_feats, prev_img = map(to_weight_type, controlnet_args)

        metas = {}
        for key, meta in sensor_metas.items():
            metas[key] = [meta]

        controlnet_kwargs = {
            "camera_param": camera_param, "ref_front_trans_mat": ref_front_trans_mat,
            "ref_rear_trans_mat": ref_rear_trans_mat, "prev_trans_mat": prev_trans_mat,
            "proj_conds": proj_conds, "meta_data": metas, "ref_front_img": ref_front_img,
            "ref_rear_img": ref_rear_img, "prev_ego_feats": prev_ego_feats, "prev_img": prev_img
        }

        print(f"  [生成] 运行diffusion pipeline...")
        image: BEVStableDiffusionPipelineOutput = self.pipeline(
            prompt=self.caption,
            height=self.diffusion_size[0],
            width=self.diffusion_size[1],
            generator=None,
            bev_controlnet_kwargs=None,
            **controlnet_kwargs,
            **self.cfg.runner.pipeline_param,
        )
        
        image_save: List[List[Image.Image]] = image.images
        # tmp = {
        #     "clr_map": proj_conds[:,:,18:,...][0].to(torch.float32),
        #     'depth_map': proj_conds[:,:,14,...][0].to(torch.float32),
        #     'proj_map': proj_conds[:,:,:6,...][0].to(torch.float32),
        #     'sem_map': torch.from_numpy(sem_map)
        # }

        tmp = {
            "clr_map": proj_conds[0,:,18:,...].to(torch.float32),
            'depth_map': proj_conds[0,:,14,...].to(torch.float32),
            'proj_map': proj_conds[0,:,:6,...].to(torch.float32),
            'sem_map': torch.from_numpy(sem_map)
        }
        
        save_dir = Path(path_config['gen_img_log'])
        img_wconcat_save(tmp, image_save, post_trans_cat, step, self.simulation.scenario.token, str(save_dir))
        
        image: List[PILImage.Image] = image.images[0]
        image = [self.post_trans(im) for im in image]
        
        # prepare vad_input (阶段3使用)
        vad_metas = {}
        vad_metas["lidar2img"] = sensor_metas["lidar2image"]
        vad_metas["lidar2cam"] = sensor_metas["lidar2camera"]
        vad_metas["lidar2global"] = ego2global

        print(f"  [生成] 完成")
        return image, vad_metas

    def run(self) -> RunnerReport:
        start_time = time.perf_counter()

        report = RunnerReport(
            succeeded=True, error_message=None,
            start_time=start_time, end_time=None, planner_report=None,
            scenario_name=self._simulation.scenario.scenario_name,
            planner_name=self.planner.name(),
            log_name=self._simulation.scenario.log_name,
        )

        print(f"\n{'='*80}")
        print(f"开始仿真场景: {self._simulation.scenario.scenario_name}")
        print(f"{'='*80}")

        self.simulation.callback.on_simulation_start(self.simulation.setup)
        self._initialize()
        
        ego_trajectory_log = []
        counter = 4  # 每5步生成一次
        prev_sensor_data = None
        prev_trajectory = None
        init_center = self.simulation.scenario.get_ego_state_at_iteration(0).center.array
        global_count = 0
        max_steps = 40  # 阶段2运行40步
        
        print(f"\n开始仿真循环（最大步数：{max_steps}，每5步生成图像）...")
        
        while self.simulation.is_simulation_running():
            if global_count >= max_steps:
                print(f"\n达到最大步数限制({max_steps})，停止仿真")
                break
                
            self.simulation.callback.on_step_start(self.simulation.setup, self.planner)
            planner_input = self.simulation.get_planner_input()
            logger.debug("Simulation iterations: %s" % planner_input.iteration.index)

            true_traj = get_traj(
                planner_input.history.ego_state_buffer[-1],
                planner_input.history.ego_state_buffer
            )

            # get sensor data
            if counter == 4:
                if self.simulation._time_controller.get_iteration().index == 0:
                    # 第一步：使用GT
                    ego_state = planner_input.history.ego_state_buffer[-1]
                    sensor_data = self.simulation.scenario.get_sensors_at_iteration(0, self.camera_channels)
                    sensor_data = [sensor_data.images[ch].as_pil for ch in self.camera_channels]
                    print(f"步骤 {global_count:3d}: 使用GT图像")
                    
                    # 保存GT图像
                    save_dir = Path(path_config['gen_img_log'])
                    gt_dir = save_dir / f"step_{0:03d}" / "gt_views"
                    gt_dir.mkdir(parents=True, exist_ok=True)
                    camera_names = ['CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_B0', 'CAM_R2', 'CAM_R1', 'CAM_R0']
                    for idx, img in enumerate(sensor_data):
                        gt_save_path = gt_dir / f"{camera_names[idx]}.jpg"
                        img.save(gt_save_path)
                    print(f"  ✓ 保存GT图像: {gt_dir}/ (8张)")
                else:
                    # 生成图像
                    print(f"步骤 {global_count:3d}: 生成图像...")
                    sensor_data, vad_metas = self.get_diffused_img(planner_input, prev_sensor_data, planner_input.iteration.index)

                prev_sensor_data = sensor_data
                counter = 0

                self._simulation.callback.on_planner_start(self.simulation.setup, self.planner)
                trajectory = true_traj
                prev_trajectory = trajectory
                self._simulation.callback.on_planner_end(self.simulation.setup, self.planner, trajectory)
            else:
                counter += 1
                trajectory = prev_trajectory
                ego_state = planner_input.history.ego_state_buffer[-1]
                current_pos = ego_state.center.array - init_center
                print(f"步骤 {global_count:3d}: 位置=({current_pos[0]:6.2f}, {current_pos[1]:6.2f})")
                
            self.simulation.propagate(trajectory)
            ego_state = planner_input.history.ego_state_buffer[-1]
            ego_trajectory_log.append(ego_state.center.array)

            global_count += 1
            self.simulation.callback.on_step_end(self.simulation.setup, self.planner, self.simulation.history.last())

            current_time = time.perf_counter()
            if not self.simulation.is_simulation_running():
                report.end_time = current_time

        self.simulation.callback.on_simulation_end(self.simulation.setup, self.planner, self.simulation.history)

        # 保存轨迹
        ego_trajectory_log = np.array(ego_trajectory_log)
        
        plt.figure(figsize=(10, 8))
        plt.plot(ego_trajectory_log[:, 0], ego_trajectory_log[:, 1], 
                marker='o', markersize=3, label='Ego Trajectory', linewidth=2)
        plt.scatter(ego_trajectory_log[0, 0], ego_trajectory_log[0, 1], 
                   c='green', s=100, label='Start', zorder=5)
        plt.scatter(ego_trajectory_log[-1, 0], ego_trajectory_log[-1, 1], 
                   c='red', s=100, label='End', zorder=5)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(fontsize=12)
        plt.title(f"Ego Vehicle Trajectory (Stage2) - {self._simulation.scenario.scenario_name}", fontsize=14)
        plt.xlabel("X (m)", fontsize=12)
        plt.ylabel("Y (m)", fontsize=12)
        
        output_dir = Path("./stage2_output")
        output_dir.mkdir(exist_ok=True)
        traj_file = output_dir / f"trajectory_{self._simulation.scenario.token}.png"
        plt.savefig(traj_file, dpi=150, bbox_inches='tight')
        print(f"\n轨迹可视化已保存至: {traj_file}")
        plt.close()

        planner_report = self.planner.generate_planner_report()
        report.planner_report = planner_report
        
        end_time = time.perf_counter()
        print(f"\n仿真完成！总耗时: {end_time - start_time:.2f}秒")
        print(f"共执行 {global_count} 步，生成 {global_count//5} 次图像")
        print(f"{'='*80}\n")

        return report


# 主程序
if __name__ == "__main__":
    print("\n开始运行NuPlan仿真（阶段2：集成Diffusion）...\n")
    
    for idx, scenario in enumerate(scenarios[:1]):
        print(f"\n处理场景 {idx+1}/{len(scenarios[:1])}: {scenario.token}")
        
        simulation_time_controller = instantiate(cfg.simulation_time_controller, scenario=scenario)
        ego_controller = instantiate(cfg.ego_controller, scenario=scenario)
        observations = build_observations(cfg.observation, scenario=scenario)
        
        simulation_setup = SimulationSetup(
            time_controller=simulation_time_controller,
            observations=observations,
            ego_controller=ego_controller,
            scenario=scenario,
        )
        
        simulation = Simulation(
            simulation_setup=simulation_setup,
            callback=MultiCallback([]),
            simulation_history_buffer_duration=cfg.simulation_history_buffer_duration,
        )
        
        planner = build_planners(cfg.planner, scenario)[0]

        runner = DiffusionSimulationRunner(
            cfg.diffusion, simulation, planner
        )
        
        try:
            report = runner.run()
            print(f"✓ 场景 {scenario.token} 运行成功")
        except Exception as e:
            print(f"✗ 场景 {scenario.token} 运行失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("阶段2完成！")
    print("="*80)
    print("\n下一步：")
    print("1. 检查stage2_output目录中的轨迹可视化")
    print("2. 检查gen_img_log目录中的生成图像")
    print("3. 如果运行成功，可以继续阶段3（接入VAD planner）")
    print("="*80)

