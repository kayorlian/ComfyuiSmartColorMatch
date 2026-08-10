import torch
import numpy as np
import cv2

class SmartColorMatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_ref": ("IMAGE",),  # 参考图 [B, H, W, C]
                "image_gen": ("IMAGE",),  # 生成图 [B, H, W, C]
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
        batch_size = image_gen.shape[0]
        ref_batch_size = image_ref.shape[0]
        
        out_tensors = []

        mask_np_batch = None
        if ignore_mask is not None:
            mask_np_batch = ignore_mask.cpu().numpy()
            if mask_np_batch.ndim == 2:
                mask_np_batch = mask_np_batch[np.newaxis, ...]

        for i in range(batch_size):
            curr_ref_img = image_ref[i % ref_batch_size].cpu().numpy()
            curr_gen_img = image_gen[i].cpu().numpy()

            # ==========================================================
            # [核心终极修复 1]：拦截 VAE 越界值，防止 OpenCV 计算崩溃产生 NaN 噪点
            # ==========================================================
            curr_ref_img = np.clip(curr_ref_img, 0.0, 1.0)
            curr_gen_img = np.clip(curr_gen_img, 0.0, 1.0)

            # 使用 INTER_AREA 避免缩小产生的摩尔纹
            if curr_ref_img.shape[:2] != curr_gen_img.shape[:2]:
                curr_ref_img = cv2.resize(curr_ref_img, (curr_gen_img.shape[1], curr_gen_img.shape[0]), interpolation=cv2.INTER_AREA)

            # ==========================================================
            # [新增修复]：动态检测并排除参考图的黑边 (黑边通常 RGB 值接近 0)
            # 计算 RGB 颜色之和，如果极小 (< 0.05) 则认为是黑边，不参与颜色统计
            # ==========================================================
            ref_non_black_mask = np.sum(curr_ref_img, axis=-1) > 0.05

            # 屏蔽遮罩逻辑（保持你原有的逻辑）
            valid_pixels_bool = None
            if mask_np_batch is not None:
                curr_mask = mask_np_batch[i % mask_np_batch.shape[0]]
                if curr_mask.shape != curr_gen_img.shape[:2]:
                    curr_mask = cv2.resize(curr_mask, (curr_gen_img.shape[1], curr_gen_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                valid_pixels_bool = curr_mask < 0.5

            ref_lab = cv2.cvtColor(curr_ref_img, cv2.COLOR_RGB2LAB)
            gen_lab = cv2.cvtColor(curr_gen_img, cv2.COLOR_RGB2LAB)

            # 结合黑边遮罩和输入的 ignore_mask
            if valid_pixels_bool is not None:
                # 参考图需要同时满足 valid_pixels_bool 并且 不是黑边
                ref_combined_mask = valid_pixels_bool & ref_non_black_mask
                ref_valid = ref_lab[ref_combined_mask]
                gen_valid = gen_lab[valid_pixels_bool]
            else:
                # 如果没有输入 ignore_mask，仅排除参考图的黑边
                ref_valid = ref_lab[ref_non_black_mask]
                gen_valid = gen_lab.reshape(-1, 3)

            # 兜底：防止 mask 异常导致没有有效像素
            if ref_valid.size == 0 or gen_valid.size == 0:
                ref_valid = ref_lab.reshape(-1, 3)
                gen_valid = gen_lab.reshape(-1, 3)

            # 提取中位数主色调
            r_mean = np.median(ref_valid, axis=0)
            g_mean = np.median(gen_valid, axis=0)
            
            r_std  = np.std(ref_valid, axis=0) + 1e-6 
            g_std  = np.std(gen_valid, axis=0) + 1e-6

            l_channel = gen_lab[:, :, 0]
            ab_channels = gen_lab[:, :, 1:]

            # 基础颜色偏移限制
            mean_shift = r_mean[1:] - g_mean[1:]
            mean_shift = np.clip(mean_shift, -15.0, 15.0)

            l_shift = r_mean[0] - g_mean[0]
            l_shift = np.clip(l_shift, -20.0, 20.0)

            if method == "reinhard_lab":
                scale = r_std[1:] / g_std[1:]
                scale = np.clip(scale, 0.5, 1.5)
                ab_channels -= g_mean[1:]
                ab_channels *= scale
                ab_channels += (g_mean[1:] + mean_shift)
                l_channel += (l_shift * 0.5) 
                
            elif method == "mkl_neutral":
                ab_channels += mean_shift
                l_channel += (l_shift * 0.5) 

            # ==========================================================
            # [核心终极修复 2]：拦截 LAB 通道越界，防止 LAB2RGB 输出垃圾颜色
            # Float32 模式下，L 必须在 0~100，ab 必须在 -127~127 之间
            # ==========================================================
            gen_lab[:, :, 0] = np.clip(l_channel, 0.0, 100.0)
            gen_lab[:, :, 1] = np.clip(ab_channels[:, :, 0], -127.0, 127.0)
            gen_lab[:, :, 2] = np.clip(ab_channels[:, :, 1], -127.0, 127.0)

            # 6. 转回 RGB
            res_rgb = cv2.cvtColor(gen_lab, cv2.COLOR_LAB2RGB)
            
            # 等比例缩放保护色相，避免生硬截断 (Gamut Mapping)
            max_c = np.max(res_rgb, axis=2, keepdims=True)
            max_c = np.where(max_c <= 0, 1.0, max_c) # 防止除以0或负数崩溃
            res_rgb = np.where(max_c > 1.0, res_rgb / max_c, res_rgb)
            
            # 最后安全封底
            res_rgb = np.clip(res_rgb, 0.0, 1.0)
            
            # 7. 混合 (Blend)
            if blend_factor < 1.0:
                res_rgb = res_rgb * blend_factor + curr_gen_img * (1 - blend_factor)
                res_rgb = np.clip(res_rgb, 0.0, 1.0)

            img_tensor = torch.from_numpy(res_rgb)
            out_tensors.append(img_tensor)

        if len(out_tensors) > 0:
            final_output = torch.stack(out_tensors, dim=0)
        else:
            final_output = image_gen

        return (final_output,)

NODE_CLASS_MAPPINGS = {
    "SmartColorMatch": SmartColorMatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatch": "Smart Color Match (Fixed)"
}