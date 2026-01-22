import torch
import numpy as np
import cv2
import gc

class SmartColorMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_ref": ("IMAGE",),  # [B, H, W, C]
                "image_gen": ("IMAGE",),  # [B, H, W, C]
                "method": (["mkl_neutral", "reinhard_lab"],),
                "blend_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "ignore_mask": ("MASK",), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "Image/Color"

    def match_color(self, image_ref, image_gen, method, blend_factor, ignore_mask=None):
        # 1. 基础参数检查
        batch_size, target_h, target_w, _ = image_gen.shape
        out_tensor = torch.empty_like(image_gen) # 预分配输出内存

        # 2. 预处理 Mask (一次性处理)
        mask_np_all = None
        if ignore_mask is not None:
            # 确保 Mask 在 CPU 并转为 Numpy
            mask_np_all = ignore_mask.cpu().numpy()
            # 处理 Mask 维度 [B, H, W]
            if mask_np_all.ndim == 2:
                mask_np_all = mask_np_all[np.newaxis, :, :]
            # 调整 Mask 大小以匹配图像
            if mask_np_all.shape[1:] != (target_h, target_w):
                # 简单的循环 resize，通常 mask batch 不大
                resized_masks = []
                for i in range(mask_np_all.shape[0]):
                    m = cv2.resize(mask_np_all[i], (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    resized_masks.append(m)
                mask_np_all = np.array(resized_masks)

        # 3. 循环处理 Batch 中的每一张图
        for b in range(batch_size):
            try:
                # --- 获取当前帧数据 (使用 clone 确保内存连续且独立) ---
                # 获取参考图并调整大小
                ref_t = image_ref[b % image_ref.shape[0]].unsqueeze(0).permute(0, 3, 1, 2) # [1, C, H, W]
                if ref_t.shape[2] != target_h or ref_t.shape[3] != target_w:
                    ref_t = torch.nn.functional.interpolate(ref_t, size=(target_h, target_w), mode="area")
                ref_t = ref_t.squeeze(0).permute(1, 2, 0) # [H, W, C]
                
                # 转 Numpy (uint8)
                ref_np = (ref_t.cpu().numpy() * 255).astype(np.uint8)
                gen_np = (image_gen[b].cpu().numpy() * 255).astype(np.uint8)

                # 处理当前 Mask
                valid_pixels_bool = None
                if mask_np_all is not None:
                    # 获取对应 mask，如果没有对应 index 则取最后一个
                    idx = min(b, mask_np_all.shape[0] - 1)
                    valid_pixels_bool = mask_np_all[idx] < 0.5
                else:
                    valid_pixels_bool = np.ones((target_h, target_w), dtype=bool)

                # --- 核心颜色迁移逻辑 ---
                ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
                gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB).astype(np.float32)

                ref_valid = ref_lab[valid_pixels_bool]
                gen_valid = gen_lab[valid_pixels_bool]

                # 避免空 mask 导致报错
                if ref_valid.size == 0: ref_valid = ref_lab.reshape(-1, 3)
                if gen_valid.size == 0: gen_valid = gen_lab.reshape(-1, 3)

                r_mean = np.mean(ref_valid, axis=0)
                r_std  = np.std(ref_valid, axis=0) + 1e-5
                g_mean = np.mean(gen_valid, axis=0)
                g_std  = np.std(gen_valid, axis=0) + 1e-5

                res_lab = gen_lab # 原地复用变量名指向
                if method == "reinhard_lab":
                    scale = r_std[1:] / g_std[1:]
                    centered = gen_lab[:, :, 1:] - g_mean[1:]
                    # 原地修改 res_lab 的通道
                    res_lab[:, :, 1:] = centered * scale + r_mean[1:]
                elif method == "mkl_neutral":
                    res_lab[:, :, 1:] = (gen_lab[:, :, 1:] - g_mean[1:]) + r_mean[1:]

                # 转回 RGB
                res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
                res_rgb = cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB)

                # --- 混合并写入输出 ---
                if blend_factor >= 1.0:
                    final_rgb = res_rgb
                elif blend_factor <= 0.0:
                    final_rgb = gen_np
                else:
                    final_rgb = cv2.addWeighted(res_rgb, blend_factor, gen_np, 1.0 - blend_factor, 0)

                # 关键优化：原地除法，减少内存峰值
                # 先转 Tensor float32
                frame_tensor = torch.from_numpy(final_rgb).float()
                # 写入到预分配的 out_tensor 中
                out_tensor[b] = frame_tensor

            except Exception as e:
                print(f"SmartColorMatch Error on batch {b}: {e}")
                out_tensor[b] = image_gen[b] # 出错则返回原图

            finally:
                # 显式删除大数组
                # 使用 locals() 检查变量是否存在，避免 UnboundLocalError
                vars_to_del = ['ref_np', 'gen_np', 'ref_lab', 'gen_lab', 'res_lab', 'res_rgb', 'final_rgb', 'ref_t', 'frame_tensor']
                for v in vars_to_del:
                    if v in locals():
                        del locals()[v]

        # 4. 统一归一化 (In-place operation)
        # 这一步比 copy 后除以 255 更省内存
        out_tensor.div_(255.0)

        # 建议：仅在大 Batch 或极高分辨率时才强制 GC，否则影响性能
        gc.collect() 
        
        return (out_tensor,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SmartColorMatch": SmartColorMatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatch": "Smart Color Match (Masked)"
}
