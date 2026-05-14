import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image, ImageFile
import debugpy

# ========= nuplanDataset =========
from dwm.datasets.nuplan import NuPlanDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========= 路径硬编码（与 cfg 一致）=========
NUSC_ROOT = "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/ggearth_files/nuplan-test/nuPlan"
train_data_pkl = "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/ggearth_files/nuplan-test/mini_infos_train.pkl"
val_data_pkl = "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/ggearth_files/nuplan-test/mini_infos_val.pkl"
IMG_DESC_PATH = "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/ggearth_files/nuplan-test/nuplan_scene.json"

# ========= transform（最小实现）=========
resize_to_tensor = T.Compose([
    T.Resize((256, 448)),  # (H, W)
    T.ToTensor()
])

def _apply_nested_images(pil_nested):
    """
    输入：nested list 形状 [T][V]，元素为 PIL.Image 或 None
    输出：同结构，但元素为 Tensor(C,H,W)，None -> 黑图占位
    """
    if pil_nested is None:
        return None
    out = []
    for row in pil_nested:
        row_t = []
        for im in row:
            if isinstance(im, Image.Image):
                row_t.append(resize_to_tensor(im))
            else:
                row_t.append(torch.zeros(3, 256, 448))  # None -> black
        out.append(row_t)
    return out

def _shape_nested(nested):
    if not nested:
        return "[]"
    Tn = len(nested)
    Vn = len(nested[0]) if Tn else 0
    x = nested[0][0] if (Tn and Vn) else None
    return f"[T={Tn}, V={Vn}, {tuple(x.shape)}]" if torch.is_tensor(x) else f"[T={Tn}, V={Vn}]"

# ========= base dataset =========
def make_base_ds(train=False):
    data_pkl = train_data_pkl if train else val_data_pkl
    return NuPlanDataset(
        "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/ggearth_files/nuplan-test/mini_sensors",
        pkl_path=data_pkl,
        cache_root='/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/ggearth_files/nuplan-test/mini_cache',
        dataset_root='/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/ggearth_files/nuplan-test/nuPlan/plan_data/mini',
        map_root='/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/ggearth_files/nuplan-test/nuPlan/maps',
        sequence_length=30,
        fps_stride_tuples=[(2, 0.5)],
        sensor_channels=['CAM_L1','CAM_L0','CAM_F0','CAM_R0','CAM_R1','CAM_R2','CAM_B0','CAM_L2'],
        enable_synchronization_check=True,    
    )

# ========= adapter：复刻 cfg transform/pop（最小实现）=========
class SimpleDatasetAdapter(torch.utils.data.Dataset):
    """
    - images -> vae_images: resize + ToTensor（None -> 黑图）
    - image_description -> clip_text（如果有）
    - pop: images / lidar_points / image_description
    """
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = dict(self.base_ds[idx])  # 拷贝，避免 side-effect

        if "images" in sample:
            sample["vae_images"] = _apply_nested_images(sample["images"])

        if "image_description" in sample:
            sample["clip_text"] = sample["image_description"]

        for k in ("images", "lidar_points", "image_description"):
            sample.pop(k, None)

        return sample

# ========= collate：batch_size=1 直接返回 =========
def collate_ignore_clip_text(batch):
    assert len(batch) == 1, "建议 batch_size=1（与 cfg 一致）"
    return batch[0]

# ========= 打印 =========
def pretty_print_sample(sample, max_show_time=2):
    print("keys:", list(sample.keys()))
    if "vae_images" in sample:
        print("vae_images:", _shape_nested(sample["vae_images"]))
    if "pts" in sample:
        pts = sample["pts"]
        print("pts:", tuple(pts.shape), "first/last(ms):", float(pts[0]), float(pts[-1]))
    if "fps" in sample:
        print("fps:", float(sample["fps"]))
    if "scene" in sample:
        print("scene:", sample["scene"])
    if "sample_data" in sample:
        x0 = sample["sample_data"][0]
        print("sample_data[0] keys:", list(x0.keys())[:12])

    if "clip_text" in sample:
        Tlen = len(sample["clip_text"])
        Vlen = len(sample["clip_text"][0]) if Tlen else 0
        print(f"clip_text: [T={Tlen}, V={Vlen}] example:")
        for t in range(min(max_show_time, Tlen)):
            ex = sample["clip_text"][t][0] if Vlen else ""
            print("  t=", t, " -> ", ex[:120])

# ========= main =========
def main():
    debugpy.listen(("0.0.0.0", 9876))
    print("[debugpy] listening on 0.0.0.0:9876, waiting for VS Code to attach...")
    debugpy.wait_for_client()
    print("[debugpy] attached")

    base_ds = make_base_ds(train=True)
    ds = SimpleDatasetAdapter(base_ds)

    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=4,
        prefetch_factor=1,
        shuffle=False,
        persistent_workers=True,
        collate_fn=collate_ignore_clip_text,
        pin_memory=False,
    )

    print("Dataset items:", len(ds))
    hit_samples = 0

    for i, batch in enumerate(loader):
        if i % 50 == 0:
            print(f"\n==== sample {i} ====")
            pretty_print_sample(batch)

        vals = batch.get("ref_quality", None)
        if vals is None:
            continue

        # vals 可能是 tensor / list / np，稳处理
        if torch.is_tensor(vals):
            nz = vals[vals != -1]
            hit = int(nz.numel())
            total = int(vals.numel())
            examples = nz[:10].tolist()
        else:
            vals_list = list(vals)
            nz = [v for v in vals_list if v != -1]
            hit, total = len(nz), len(vals_list)
            examples = nz[:10]

        if hit > 0:
            hit_samples += 1
            print(f"[HIT] sample #{i}: found {hit}/{total} values != -1")
            print("      examples:", examples)

        if i % 200 == 0:
            print(f"[progress] seen {i+1}/{len(ds)}; hits so far = {hit_samples}")

    print(f"[RESULT] samples with any ref_quality != -1: {hit_samples}")

if __name__ == "__main__":
    main()
