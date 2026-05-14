import torch



x = [k for k in range(150) if k not in set(range(3, 151, 3))]

print(x)


class DFIdxTester:
    def __init__(self, L=16, T=32, clear_ref=0, S=2, S_mode="manual", device="cpu"):
        """
        S_mode:
          - "derive": S = T // (L - clear_ref)  (与你代码一致)
          - "manual": 使用传入的 S
        """
        self.device = device
        self.L = int(L)
        self.T = int(T)
        self.clear_ref = int(clear_ref)
        self.S = int(S)
        assert S_mode in ("manual", "derive")
        self.S_mode = S_mode

    def _get_S(self):
        if self.S_mode == "derive":
            return self.T // (self.L - self.clear_ref)
        return self.S

    # ===== 内层：保持你现在的 idx 计算“一字不改”=====
    def inference_pipeline(
        self,
        latent_shape,
        start_timestep: int = 0,
        stop_timestep: int | None = None,
        take_time: int = 0,
        frozen_ref_count: int = 0,
        verbose_each_i: bool = True,   # True: 打印每个 i 的 idx; False: 只打印 first/last
    ):
        L = latent_shape[1]
        T = self.T
        S = self._get_S()
        clear_reference_frame_count = self.clear_ref

        j = torch.arange(L, device=self.device)
        j_eff = torch.clamp(j - clear_reference_frame_count, min=0)

        if stop_timestep is None:
            stop_timestep = T

        idx_first = None
        idx_last  = None

        print(f"[CALL] start={start_timestep}, stop={stop_timestep}, take_time={take_time}, "
              f"frozen_ref_count={frozen_ref_count}, (L={L}, T={T}, S={S}, clear_ref={clear_reference_frame_count})")

        for i in range(start_timestep, stop_timestep):
            # ---- 下面完全照抄你的逻辑 ----
            base = i - take_time * S
            inner = torch.clamp(base - j_eff * S, min=0)
            idx = torch.minimum(inner, torch.full_like(inner, base))
            max_idx = T - 1
            idx = torch.clamp(idx, 0, max_idx)

            if frozen_ref_count > 0:
                clean_idx = max_idx
                idx[:frozen_ref_count] = clean_idx
            # ---- 以上完全照抄你的逻辑 ----

            if idx_first is None:
                idx_first = idx.clone()
            idx_last = idx.clone()

            if verbose_each_i:
                print(f"  i={i:4d} base={base:4d} idx={idx.cpu().tolist()}")

        if not verbose_each_i:
            print("  idx_first:", None if idx_first is None else idx_first.cpu().tolist())
            print("  idx_last :", None if idx_last  is None else idx_last.cpu().tolist())

        return {
            "idx_first": None if idx_first is None else idx_first.cpu().tolist(),
            "idx_last":  None if idx_last  is None else idx_last.cpu().tolist(),
            "L": L, "T": T, "S": S, "clear_ref": clear_reference_frame_count,
        }

    # ===== 外层：normal phase + tail flush phase（只负责传参，不改内层）=====
    def run_test(
        self,
        dataset_ref_left: int = 0,
        normal_iters: int = 3,
        verbose_each_i: bool = False,
        # normal phase 参数（对应你外层那段）
        normal_start=None,  # 默认用 T-S
        normal_stop=None,   # 默认用 T
        normal_take_time: int = 0,
        # tail flush 参数（对应你现在尾刷那段）
        flush_steps=None,   # 默认 min(L-clear_ref, L-1)
        tail_start_fn=None, # 默认用你现在那条: T + (2*k-2)*S
        tail_stop_fn=None,  # 默认用你现在那条: T + (2*k-1)*S
        tail_frozen_fn=None # 默认 frozen_k = min(dataset_ref_left + (k-1), L)
    ):
        L, T, S, C = self.L, self.T, self._get_S(), self.clear_ref
        latent_shape = (1, L, 1, 1, 1, 1)

        if normal_start is None: normal_start = T - S
        if normal_stop  is None: normal_stop  = T

        if flush_steps is None:
            flush_steps = L - C

        if tail_start_fn is None:
            tail_start_fn = lambda k: T + (2 * k - 1) * S
        if tail_stop_fn is None:
            tail_stop_fn  = lambda k: T + (2 * k) * S
        if tail_frozen_fn is None:
            tail_frozen_fn = lambda k: min(dataset_ref_left + k, L)
        print("\n====================")
        print("PARAMS")
        print(f"  L={L}, T={T}, S={S} ({self.S_mode}), clear_ref={C}, dataset_ref_left={dataset_ref_left}")
        print("====================\n")

        print("=== NORMAL PHASE ===")
        for it in range(normal_iters):
            print(f"\n[NORMAL iter={it}]")
            self.inference_pipeline(
                latent_shape=latent_shape,
                start_timestep=normal_start,
                stop_timestep=normal_stop,
                take_time=normal_take_time,
                frozen_ref_count=dataset_ref_left,
                verbose_each_i=verbose_each_i,
            )

        print("\n=== TAIL FLUSH PHASE ===")
        for k in range(1, flush_steps):
            frozen_k = tail_frozen_fn(k)
            start_k = tail_start_fn(k)
            stop_k  = tail_stop_fn(k)

            print(f"\n[FLUSH k={k}] frozen_k={frozen_k}")
            self.inference_pipeline(
                latent_shape=latent_shape,
                start_timestep=start_k,
                stop_timestep=stop_k,
                take_time=k,
                frozen_ref_count=frozen_k,
                verbose_each_i=verbose_each_i,
            )


if __name__ == "__main__":
    # ====== 你可以随便改这些 toy 参数 ======
    tester = DFIdxTester(
        L=16,
        T=32,
        clear_ref=0,
        S_mode="derive",   # 改成 "derive" 就按 T//(L-clear_ref) 自动算 S
        device="cpu"
    )

    # 示例：视频16, ref15
    tester.run_test(
        dataset_ref_left=0,
        normal_iters=1,
        verbose_each_i=True,  # True 会打印每个 i 的 idx（更啰嗦）
    )