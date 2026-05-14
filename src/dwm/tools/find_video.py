# diff_bottom256_max.py
import os, glob, cv2, numpy as np

# ===== 硬编码区 =====
DIR_A = "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/ggearth/OpenDWM/output/geo_preview/preview"   # 文件夹A
DIR_B = "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/ggearth/OpenDWM/output/nogeo_preview12/preview"   # 文件夹B（与A中文件名一一对应）
ROW_H = 256                   # 取最后一行的高度
TOPK  = 10                    # 想看差异最大的前K个
# ====================

EXTS = {".mp4",".avi",".mov",".mkv",".ts",".m4v",".flv",".wmv"}

def ahash(img, size=16):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    m = g.mean()
    return (g > m).astype(np.uint8).flatten()

def read_bottom_row256_first_frame(p, row_h=256):
    cap = cv2.VideoCapture(p); ok, f = cap.read(); cap.release()
    if not ok or f is None: return None
    h = f.shape[0]
    rh = min(row_h, h)  # 防越界
    return f[h - rh : h, :, :]

def video_list(d):
    return {os.path.basename(p): p
            for p in glob.glob(os.path.join(d, "*"))
            if os.path.splitext(p)[1].lower() in EXTS}

if __name__ == "__main__":
    A = video_list(DIR_A)
    B = video_list(DIR_B)
    common = sorted(set(A) & set(B))
    if not common:
        print("两个文件夹没有同名视频～"); raise SystemExit

    diffs = []
    for name in common:
        a = read_bottom_row256_first_frame(A[name], ROW_H)
        b = read_bottom_row256_first_frame(B[name], ROW_H)
        if a is None or b is None: 
            continue
        da, db = ahash(a), ahash(b)
        d = (da != db).sum()      # 汉明距离：0~(16*16)=256
        diffs.append((d, name))

    diffs.sort(reverse=True)      # 差异从大到小
    print(f"共比较 {len(diffs)} 个匹配视频，差异最大的Top-{min(TOPK,len(diffs))}:")
    for d, n in diffs[:TOPK]:
        print(f"{d:3d}  {n}")

    if diffs:
        print("\n最不相似（差异最大）的文件：", diffs[0][1])
