import numpy as np
import numpy.typing as npt
from typing import Any, Optional, Tuple, Type, List
import logging
import time
import torch
import torchvision
import h5py
import os
from pathlib import Path
import gc
from datetime import datetime

from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInput
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.observation.observation_type import CameraChannel
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.camera import Camera
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)

from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db

from magicdrive.pipeline.pipeline_prevproj import StableDiffusionPrevProjPipeline
from magicdrive.pipeline.pipeline_bev_controlnet import (
    BEVStableDiffusionPipelineOutput,
)
from magicdrive.dataset.pipeline_utils import one_hot_decode_proj
from PIL import Image as PILImage
import cv2
from .vad_planner import VADPlannerInput

from magicdrive.nuplan_sim.common import (
    rotate_round_z_axis, 
    resize_img, 
    get_transmat_for_lidarpc_token_from_db,
    obtain_sensor2top,
    get_aug_mat
)
from magicdrive.nuplan_sim.proj_ctrls_utils import (
    get_projected_map,
    get_projected_bboxes,
)
from magicdrive.runner.utils import concat_6_views
from magicdrive.misc.test_utils import build_pipe
import pickle


logger = logging.getLogger(__name__)

class DiffusionSimulationRunner(SimulationRunner):
    def __init__(
        self, 
        cfg,
        simulation: Simulation, 
        planner: AbstractPlanner, 
        # pipeline: StableDiffusionPrevProjPipeline,
        # db_record: NuPlanDB,
        # numap: NuPlanMap,
        # weight_dtype=torch.bfloat16,
        kernel_size=(3, 3),
        sigma=1.0,
        patch_radius=100,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        ref_mean=[0.485, 0.456, 0.406],
        ref_std=[0.229, 0.224, 0.225],
        im_size=(1080, 1920),
        diffusion_size=(224, 400),
        resize_lim=[1/4.8, 1/4.8],
        use_timestamp_sensor=True,
        save_gt_bboxes=True,
    ):
        super().__init__(simulation, planner)
        self.cfg = cfg
        # self.pipeline = pipeline
        self.camera_channels = [
            CameraChannel.CAM_F0,
            CameraChannel.CAM_L0,
            CameraChannel.CAM_L1,
            CameraChannel.CAM_L2,
            CameraChannel.CAM_B0,
            CameraChannel.CAM_R2,
            CameraChannel.CAM_R1,
            CameraChannel.CAM_R0,
        ]
        # self.numap = numap
        # self.db_record = db_record
        self.diffusion_size = diffusion_size
        self.resize_lim = resize_lim
        self.patch_radius = patch_radius
        self.im_size = im_size
        # self.weight_dtype = weight_dtype
        # cache_file = list(cfg.dataset.dataset_cache_file)[1]
        cache_file = cfg.simulation_cache_file
        if cache_file and os.path.isfile(cache_file):
            logging.info(f"using data cache from: {cache_file}")
            # load to memory and ignore all possible changes.
            self.proj_cond_cache = cache_file
        else:
            self.proj_cond_cache = None
        self.prev_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.ref_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=ref_mean, std=ref_std),
            ]
        )
        self.post_trans = torchvision.transforms.Compose( 
            [
                torchvision.transforms.Resize(
                    (1075, 1920),
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                ),
                torchvision.transforms.Pad(
                    [0,5,0,0]
                ),
        ])

        self.polygon_layer_names = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        ]
        self.line_layer_names = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
        ]
        self.obj_classes = [
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.BARRIER,
            TrackedObjectType.CZONE_SIGN
        ]
        self.use_timestamp_sensor = use_timestamp_sensor
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.saving_root = Path.joinpath(Path("/mnt/hcufs/home/youjunqi/GENAD2/a_lot_of_log_imgs/"), current_time, self.simulation.scenario.scenario_name)
        os.makedirs(self.saving_root, exist_ok=True)
        self.save_gt_bboxes = save_gt_bboxes
        self.all_gt_bboxes = dict()
        self.all_pred_results = dict()
        self.all_vad_inputs = dict()

    def clean_up(self) -> None:
        # self._planner.model.to("cpu")
        # self.pipeline = None
        del self._planner
        del self.pipeline
        torch.cuda.empty_cache()
        gc.collect()

    def _initialize(self) -> None:
        """
        Initialize the planner
        """
        # Execute specific callback
        self._simulation.callback.on_initialization_start(self._simulation.setup, self.planner)

        # Initialize Planner
        self.planner.initialize(self._simulation.initialize())

        # Execute specific callback
        self._simulation.callback.on_initialization_end(self._simulation.setup, self.planner)

        self.pipeline, self.weight_dtype = build_pipe(self.cfg, device="cuda")

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

    def get_sensor_metas(self, ego2global):
        info = {"cam":{}}
        for cam in self.camera_channels:
            cam_db: Camera = self.db_record.camera.select_one(channel = cam.value)
            cam_info = obtain_sensor2top(
                cam_db, self.db_record.lidar[0], ego2global
            )
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
            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(
                camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T)
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            ret["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            distortion = ret["distortion"][-1]
            cam_intrin = camera_intrinsics[:3, :3]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_intrin, distortion, (w, h), 1, (w, h))
            new_cam_mat_pad = np.pad(new_camera_matrix, ((0, 1), (0, 1)), mode='constant')
            new_cam_mat_pad[3, 3] = 1
            ret["camera_intrinsics"].append(new_cam_mat_pad)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            ret["lidar2image"].append(lidar2image)
                        # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            ret["camera2lidar"].append(camera2lidar)

            # sensor2ego translation&rotation
            ret["sensor2ego_translation"].append(camera_info["sensor2ego_translation"])
            ret["sensor2ego_rotation"].append(camera_info["sensor2ego_rotation"])
        return ret

    def prepare_ego_coords(self):
        ego_coords = []
        for it in range(self.simulation.scenario.get_number_of_iterations()):
            ego_state = self.simulation.scenario.get_ego_state_at_iteration(it)
            ego_coords.append(ego_state.rear_axle.point.array)
        self.scene_ego_coords = np.stack(ego_coords)

    def get_ref_index(self, ego_coord, velo):
        vec = self.scene_ego_coords - ego_coord
        score = np.sum(vec * velo, axis=-1)
        # get the index of minimum positive score and maximum negtive score

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
        sensors = self.simulation.scenario.get_sensors_at_iteration(idx, self.camera_channels)
        imgs = [sensors.images[ch].as_pil for ch in self.camera_channels]
        if len(imgs) != 8:
            for i in range(5):
                sensors = self.simulation.scenario.get_sensors_at_iteration(idx+i, self.camera_channels)
                imgs = [sensors.images[ch].as_pil for ch in self.camera_channels]
                if len(imgs) == 8:
                    break
        # ref_vis = [imgs[3], imgs[2], imgs[1], imgs[0], imgs[-1], imgs[-2], imgs[-3], imgs[-4]]
        # ref_vis = concat_6_views(ref_vis, oneline=True)
        # ref_vis.save(f"gen_img_log/ref_vis_{num}_{step}.png")
        return torch.stack([
            self.ref_transform(resize_img(img, self.resize_lim[0], self.diffusion_size))
            for img in imgs
        ])

    def get_ego2global(self, ego_state, initial_ego2global):
        trans_z = initial_ego2global[2,3]
        ego2global = ego_state.rear_axle.as_matrix_3d()
        ego2global[2,3] = trans_z
        return ego2global

    def get_object_bboxes(self, observation, ego_state):
        tracked_objs: TrackedObjects = observation.tracked_objects
        ego2global = ego_state.rear_axle.as_matrix()
        gt_bboxes = []
        gt_labels = []
        for obj_cls in self.obj_classes:
            obj_list = tracked_objs.get_tracked_objects_of_type(obj_cls)
            if len(obj_list) == 0:
                continue
            box_list = [obj.box for obj in obj_list]
            locs_2d = np.array([b.center.point.array for b in box_list]).reshape(-1, 2)
            locs_2d = locs_2d - ego2global[:2, 2].reshape(-1, 2)
            locs_2d = locs_2d @ np.linalg.inv(ego2global[:2, :2]).T
            locs = np.concatenate([locs_2d, np.zeros_like(locs_2d)[..., :1]], axis=-1)
            dims = np.array([[b.width, b.length, b.height] for b in box_list]).reshape(-1, 3)
            rots = np.array([b.center.heading - ego_state.rear_axle.heading for b in box_list]).reshape(-1, 1)
            boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

            labels = int(obj_cls) * np.ones(len(boxes))
            gt_bboxes.append(boxes)
            gt_labels.append(labels)
        if len(gt_bboxes) != 0:
            gt_bboxes = np.concatenate(gt_bboxes, axis=0)
            gt_bboxes_mmcv = LiDARInstance3DBoxes(
                gt_bboxes, box_dim=gt_bboxes.shape[-1], origin=(0.5, 0.5, 0)
            )
            gt_labels = np.concatenate(gt_labels, axis=0)
            gt_bboxes = np.concatenate([gt_bboxes, np.zeros_like(gt_bboxes[..., :2])], axis=-1)
            gt_bboxes = np.concatenate([gt_bboxes, gt_labels.reshape(-1,1)], axis=-1)
        else: 
            gt_bboxes_mmcv = []
        return gt_bboxes_mmcv, gt_bboxes, gt_labels

    def run(self) -> RunnerReport:
        start_time = time.perf_counter()

        # Initialize reports for all the simulations that will run
        report = RunnerReport(
            succeeded=True,
            error_message=None,
            start_time=start_time,
            end_time=None,
            planner_report=None,
            scenario_name=self._simulation.scenario.scenario_name,
            planner_name=self.planner.name(),
            log_name=self._simulation.scenario.log_name,
        )

        # Execute specific callback
        self.simulation.callback.on_simulation_start(self.simulation.setup)

        # Initialize all simulations
        self._initialize()

        counter = 4
        prev_sensor_data = None
        prev_trajectory = None
        init_center = self.simulation.scenario.get_ego_state_at_iteration(0).center.array
        while self.simulation.is_simulation_running():
            # Execute specific callback
            self.simulation.callback.on_step_start(self.simulation.setup, self.planner)

            # Perform step
            planner_input = self.simulation.get_planner_input()
            logger.debug("Simulation iterations: %s" % planner_input.iteration.index)

            # get sensor data according to planner input
            if counter == 4:
                if self.simulation._time_controller.get_iteration().index == 0 or self.use_timestamp_sensor:
                    # provide gt sensor data at the first step
                    sensor_data, vad_metas = self.get_timestamp_sensor_at_iteration(planner_input, planner_input.iteration.index)
                    # sensor_data = self.get_diffused_img(planner_input, sensor_data)
                else:
                    sensor_data, vad_metas = self.get_diffused_img(planner_input, prev_sensor_data, planner_input.iteration.index)
                    # sensor_data = None
                vad_input = VADPlannerInput(sensor_data, vad_metas)
                prev_sensor_data = sensor_data
                counter = 0

                # Execute specific callback
                self._simulation.callback.on_planner_start(self.simulation.setup, self.planner)

                # Plan path based on all planner's inputs
                trajectory, vad_outputs, all_vad_metas = self.planner.compute_planner_trajectory(planner_input, vad_input)
                if self.simulation._time_controller.get_iteration().index != 0:
                    self.all_pred_results[f"{self.simulation.scenario.token}_{planner_input.iteration.index}"] = vad_outputs[0]
                    gt_bboxes, gt_bboxes_py, gt_labels = self.get_object_bboxes(planner_input.history.observation_buffer[-1], planner_input.history.ego_state_buffer[-1])
                    self.all_gt_bboxes[f"{self.simulation.scenario.token}_{planner_input.iteration.index}"] = gt_bboxes_py
                    self.all_vad_inputs[f"{self.simulation.scenario.token}_{planner_input.iteration.index}"] = all_vad_metas
                # print(f"trajectory at outside: {tj}")
                prev_trajectory = trajectory
                # Propagate simulation based on planner trajectory
                self._simulation.callback.on_planner_end(self.simulation.setup, self.planner, trajectory)
            else:
                counter += 1
                # trajectory = InterpolatedTrajectory(prev_trajectory.get_sampled_trajectory()[counter:])
                trajectory = prev_trajectory
                
            self.simulation.propagate(trajectory)
            aaego_state = planner_input.history.ego_state_buffer[-1]
            # print("current_ego_velocity", aaego_state.dynamic_car_state.rear_axle_velocity_2d.array)
            # print("current_ego_state", aaego_state.center.array - init_center)

            # Execute specific callback
            self.simulation.callback.on_step_end(self.simulation.setup, self.planner, self.simulation.history.last())

            # Store reports for simulations which just finished running
            current_time = time.perf_counter()
            if not self.simulation.is_simulation_running():
                report.end_time = current_time

        # Execute specific callback
        self.simulation.callback.on_simulation_end(self.simulation.setup, self.planner, self.simulation.history)
        os.makedirs(Path.joinpath(self.saving_root, "bboxes"), exist_ok=True)
        with open(Path.joinpath(self.saving_root, "bboxes", "gt_bboxes.pkl"), "wb") as f:
            pickle.dump(self.all_gt_bboxes, f)
        with open(Path.joinpath(self.saving_root, "bboxes", "pred_results.pkl"), "wb") as f:
            pickle.dump(self.all_pred_results, f)
        with open(Path.joinpath(self.saving_root, "bboxes", "vad_inputs.pkl"), "wb") as f:
            pickle.dump(self.all_vad_inputs, f)

        planner_report = self.planner.generate_planner_report()
        report.planner_report = planner_report

        self.clean_up()

        return report

    def get_timestamp_sensor_at_iteration(self, planner_input, iteration):
        ego_state = planner_input.history.ego_state_buffer[-1]
        # print("current_ego_state", ego_state.center.array - init_center)
        initial_ego2global = get_transmat_for_lidarpc_token_from_db(
            self.simulation.scenario._log_file, self.simulation.scenario._initial_lidar_token
        )
        ego2global = self.get_ego2global(ego_state, initial_ego2global)
        sensor_metas = self.get_sensor_metas(ego2global)
        vad_metas = {}
        vad_metas["lidar2img"] = sensor_metas["lidar2image"]
        vad_metas["lidar2cam"] = sensor_metas["lidar2camera"]
        vad_metas["lidar2global"] = ego2global
        sensor_data = self.simulation.scenario.get_sensors_at_iteration(iteration, self.camera_channels)
        sensor_data = [sensor_data.images[ch].as_pil for ch in self.camera_channels]

        if iteration != 0:
            img_vis = [sensor_data[3], sensor_data[2], sensor_data[1], sensor_data[0], sensor_data[-1], sensor_data[-2], sensor_data[-3], sensor_data[-4]]
            img_vis = concat_6_views(img_vis, oneline=True)
            img_vis.save(Path.joinpath(self.saving_root, f"img_vis_{iteration}.png"))
        return sensor_data, vad_metas

    def get_diffused_img(self, planner_input: PlannerInput, prev_sensor_data: List[PILImage.Image], step):
        assert len(planner_input.history.ego_state_buffer) > 1
        ego_state = planner_input.history.ego_state_buffer[-1]
        
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
        ref_tokens = []
        for i, index in enumerate(ref_indices):
            ref_img.append(self.get_ref_sensors(index, i, step))
            ref_ego2global = self.get_ego2global(
                self.simulation.scenario.get_ego_state_at_iteration(index), initial_ego2global
            )
            ref_trans_mats.append(torch.from_numpy(np.linalg.inv(ego2global) @ (ref_ego2global)))
            # ref_trans_mats.append(torch.ones(4))
            ref_tokens.append(
                self.simulation.scenario.get_scenario_tokens()[index]
            )
        ref_front_img, ref_rear_img = ref_img
        ref_front_trans_mat, ref_rear_trans_mat = ref_trans_mats
        # ref_front_trans_mat = ref_rear_trans_mat = torch.eye(4)
        ref_front_token, ref_rear_token = ref_tokens
        # print(ref_front_trans_mat)

        with h5py.File(self.proj_cond_cache, 'r') as cache_file:
            ref_front_attn_mask = torch.from_numpy(one_hot_decode_proj(
                cache_file["loss_mask"][ref_front_token][:], 8, 8))
            ref_rear_attn_mask = torch.from_numpy(one_hot_decode_proj(
                cache_file["loss_mask"][ref_rear_token][:], 8, 8))

        # get projected controls
        sensor_metas = self.get_sensor_metas(ego2global)
        sensor2ego_t_list = sensor_metas["sensor2ego_translation"]
        sensor2ego_r_list = sensor_metas["sensor2ego_rotation"]
        cam_intrinsics_list = sensor_metas["camera_intrinsics"]
        lidar2image_list = sensor_metas["lidar2image"]
        map_ctrls = get_projected_map(
            self.numap,
            self.patch_radius,
            8,
            (self.im_size[1], self.im_size[0]),
            self.diffusion_size,
            ego2global,
            sensor2ego_t_list,
            sensor2ego_r_list,
            cam_intrinsics_list,
            self.polygon_layer_names,
            self.line_layer_names,
        )

        cur_observation = planner_input.history.observation_buffer[-1]
        gt_bboxes, gt_bboxes_py, gt_labels = self.get_object_bboxes(cur_observation, ego_state)
        obj_ctrls, _ = get_projected_bboxes(
            gt_bboxes,
            gt_labels,
            8,
            (self.im_size[1], self.im_size[0]),
            self.diffusion_size,
            lidar2image_list,
            self.obj_classes,
            TrackedObjectType.VEHICLE,
        )


        # box_ctrls = obj_ctrls[:, 0]
        # masks = [PILImage.fromarray(mask*255) for mask in box_ctrls]
        # masks = [masks[3], masks[2], masks[1], masks[0], masks[-1], masks[-2], masks[-3], masks[-4]]
        # mask_to_showb = concat_6_views(masks, oneline=True)
        # mask_to_showb.save(f"gen_img_log/box_mask_{step}.png")

        # line_ctrls = map_ctrls[:, -2]
        # masks = [PILImage.fromarray(mask*255) for mask in line_ctrls]
        # masks = [masks[3], masks[2], masks[1], masks[0], masks[-1], masks[-2], masks[-3], masks[-4]]
        # mask_to_showl = concat_6_views(masks, oneline=True)
        # mask_to_showl.save(f"gen_img_log/map_mask_{step}.png")

        proj_conds = torch.from_numpy(np.concatenate([obj_ctrls, map_ctrls], axis=1)).float()

        camera_param = torch.cat([
                torch.from_numpy(np.stack(sensor_metas["camera_intrinsics"])[:, :3, :3]),
                torch.from_numpy(np.stack(sensor_metas["camera2lidar"])[:, :3]),  # only first 3 rows meaningful
            ], dim=-1)
        
        
        sensor_metas["img_aug_matrix"] = torch.stack([get_aug_mat(prev_sensor_data[0], self.resize_lim[0], self.diffusion_size)] * 8)
        # sensor_metas["img_aug_matrix"] = torch.stack([torch.eye(4)] * 8)
        to_weight_type = lambda x: x.to(self.weight_dtype).unsqueeze(0)

        controlnet_args = [camera_param, ref_front_trans_mat, ref_rear_trans_mat, prev_trans_mat, proj_conds, \
            ref_front_img, ref_rear_img, prev_ego_feats, prev_img, ref_front_attn_mask, ref_rear_attn_mask]

        camera_param, ref_front_trans_mat, ref_rear_trans_mat, prev_trans_mat, proj_conds, \
            ref_front_img, ref_rear_img, prev_ego_feats, prev_img, ref_front_attn_mask, ref_rear_attn_mask = map(to_weight_type, controlnet_args)

        metas = {}
        for key, meta in sensor_metas.items():
            metas[key] = [meta]

        controlnet_kwargs = {"camera_param": camera_param, "ref_front_trans_mat": ref_front_trans_mat, \
                        "ref_front_attn_mask": ref_front_attn_mask, "ref_rear_attn_mask": ref_rear_attn_mask,\
                        "ref_rear_trans_mat": ref_rear_trans_mat, "prev_trans_mat": prev_trans_mat, \
                        "proj_conds": proj_conds, "meta_data": metas, "ref_front_img": ref_front_img, \
                        "ref_rear_img": ref_rear_img, "prev_ego_feats": prev_ego_feats,  "prev_img": prev_img}
        
        image: BEVStableDiffusionPipelineOutput = self.pipeline(
            prompt=self.caption,
            height=self.diffusion_size[0],
            width=self.diffusion_size[1],
            generator=None,
            bev_controlnet_kwargs=None,
            **controlnet_kwargs,
            **self.cfg.runner.pipeline_param,
        )
        image: List[PILImage.Image] = image.images[0]
        image = [self.post_trans(im) for im in image]
        # prepare vad_input
        vad_metas = {}
        vad_metas["lidar2img"] = sensor_metas["lidar2image"]
        vad_metas["lidar2cam"] = sensor_metas["lidar2camera"]
        vad_metas["lidar2global"] = ego2global

        img_vis = [image[3], image[2], image[1], image[0], image[-1], image[-2], image[-3], image[-4]]
        img_vis = concat_6_views(img_vis, oneline=True)
        img_vis.save(Path.joinpath(self.saving_root, f"img_vis_{step}.png"))

        return image, vad_metas
