#!/usr/bin/env python3
import os
import sys
import pickle
import random
from collections import Counter
from PIL import Image
import debugpy

def main(pkl_path, out_dir=None, sample_n=20, seed=0):
    
    debugpy.listen(("0.0.0.0", 9876))
    print("[debugpy] listening on, waiting for VS Code to attach...")
    debugpy.wait_for_client()        
    print("attached")
    
    print(f"[load] {pkl_path}")
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    print(f"type(obj)={type(obj)}")

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"dict size: {len(keys)}")
        print("key sample:", keys[:5])

        type_counter = Counter(type(v).__name__ for v in obj.values())
        print("value type counts:", dict(type_counter))

        if keys and isinstance(obj[keys[0]], Image.Image):
            size_counter = Counter()
            mode_counter = Counter()
            for v in obj.values():
                if isinstance(v, Image.Image):
                    size_counter[v.size] += 1
                    mode_counter[v.mode] += 1
            print("image size counts (top 10):", size_counter.most_common(10))
            print("image mode counts:", dict(mode_counter))

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            random.seed(seed)
            take = keys if sample_n <= 0 else random.sample(keys, min(sample_n, len(keys)))
            saved = 0
            for k in take:
                v = obj[k]
                if isinstance(v, Image.Image):
                    out_path = os.path.join(out_dir, f"{k}.png")
                    if len(os.path.basename(out_path)) > 200:
                        out_path = os.path.join(out_dir, f"{saved}.png")
                    v.save(out_path)
                    saved += 1
                else:
                    with open(os.path.join(out_dir, f"{saved}.txt"), "w", encoding="utf-8") as fw:
                        fw.write(f"type={type(v)}\nrepr={repr(v)[:500]}\n")
                    saved += 1
            print(f"exported {saved} samples to: {out_dir}")

    else:
        print("Non-dict pickle. Repr head:")
        s = repr(obj)
        print(s[:1000] + ("..." if len(s) > 1000 else ""))

if __name__ == "__main__":

    main('/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/ggearth_files/nuplan-test/mini_infos_train.pkl', out_dir=None)
