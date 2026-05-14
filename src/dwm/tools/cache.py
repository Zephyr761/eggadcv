# prewarm_cache_minimal.py  —— 只缓存 PNG，不做其他变换/堆叠
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem

from dwm.datasets.nuscenes import MotionDataset

# ========= 路径硬编码（与训练一致的根目录 & 你的缓存根目录） =========
NUSC_ROOT  = "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/ggearth/dataset/nuscenes"
CACHE_ROOT = "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/ggearth/dataset/nus_cache"
os.makedirs(CACHE_ROOT, exist_ok=True)

# 用本地文件系统挂载 nuScenes 根
fs = DirFileSystem(path=NUSC_ROOT, fs=LocalFileSystem())

# ========= 只缓存所需的 Dataset（不挂任何 Adapter/transform） =========
def make_train_ds():
    return MotionDataset(
        fs=fs,
        dataset_name="interp_12Hz_trainval",
        split="train",
        sequence_length=19,
        # 和训练一致：按你的配置来；只影响采样哪些片段
        fps_stride_tuples=[(10, 1)],
        sensor_channels=[
            "LIDAR_TOP",
            "CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT",
        ],
        keyframe_only=True,
        # 预热只为出 PNG，全部关掉不必要开关，尽量省时
        enable_camera_transforms=False,
        enable_ego_transforms=False,
        enable_sample_data=False,
        _3dbox_image_settings={},   # 开 3dbox 渲染 → 命中/生成 PNG
        hdmap_image_settings={},    # 开 HD map 渲染 → 命中/生成 PNG
        image_segmentation_settings=None,
        foreground_region_image_settings=None,
        _3dbox_bev_settings=None,
        hdmap_bev_settings=None,
        image_description_settings=None,
        stub_key_data_dict=None,
    )

def make_val_ds():
    return MotionDataset(
        fs=fs,
        dataset_name="interp_12Hz_trainval",
        split="val",
        sequence_length=35,
        fps_stride_tuples=[(10, 20)],
        sensor_channels=[
            "LIDAR_TOP",
            "CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT",
        ],
        keyframe_only=True,
        enable_synchronization_check=False,  # 与训练一致
        enable_camera_transforms=False,
        enable_ego_transforms=False,
        enable_sample_data=False,
        _3dbox_image_settings={},
        hdmap_image_settings={},
        image_segmentation_settings=None,
        foreground_region_image_settings=None,
        _3dbox_bev_settings=None,
        hdmap_bev_settings=None,
        image_description_settings=None,
        stub_key_data_dict=None,
    )

# ========= 极简 collate：不做任何堆叠/拼接 =========
def collate_cache_only(_batch):
    # 目的只是触发 __getitem__ 里读/写缓存；返回 None 即可
    return None

def warm(ds, num_workers=32, desc="prewarm", batch_size=8):
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        prefetch_factor=3,
        persistent_workers=True,
        pin_memory=False,
        collate_fn=collate_cache_only,
    )
    for _ in tqdm(loader, total=len(loader), desc=desc, dynamic_ncols=True):
        # 不取用返回值，只要让 __getitem__ 跑起来写 PNG
        pass

if __name__ == "__main__":
    train_ds = make_train_ds()
    val_ds   = make_val_ds()

    print(f"Train items: {len(train_ds)}")
    print(f"Val items:   {len(val_ds)}")

    print("Prewarm train cache...")
    warm(train_ds, num_workers=42, desc="train cache", batch_size=4)

    print("Prewarm val cache...")
    warm(val_ds,   num_workers=32, desc="val cache",   batch_size=2)

    print("Done. Cached PNGs at:", CACHE_ROOT)
