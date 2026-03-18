import torch
import numpy as np
import cv2
import gc

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

        # 预处理 Mask
        mask_np_batch = None
        if ignore_mask is not None:
            mask_np_batch = ignore_mask.cpu().numpy()
            if mask_np_batch.ndim == 2:
                mask_np_batch = mask_np_batch[np.newaxis, ...]

        for i in range(batch_size):
            # 1. 准备数据：保持 Float32 (0.0 - 1.0) 精度
            curr_ref_img = image_ref[i % ref_batch_size].cpu().numpy()
            curr_gen_img = image_gen[i].cpu().numpy()

            # 确保尺寸一致 (使用 INTER_AREA 避免缩小产生的摩尔纹)
            if curr_ref_img.shape[:2] != curr_gen_img.shape[:2]:
                curr_ref_img = cv2.resize(curr_ref_img, (curr_gen_img.shape[1], curr_gen_img.shape[0]), interpolation=cv2.INTER_AREA)

            # 2. 处理 Mask
            valid_pixels_bool = None
            curr_mask = None
            if mask_np_batch is not None:
                curr_mask = mask_np_batch[i % mask_np_batch.shape[0]]
                if curr_mask.shape != curr_gen_img.shape[:2]:
                    curr_mask = cv2.resize(curr_mask, (curr_gen_img.shape[1], curr_gen_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                # < 0.5 (黑色部分) 是目标衣服区域
                valid_pixels_bool = curr_mask < 0.5

            # 3. 颜色空间转换 RGB -> LAB (Float32)
            ref_lab = cv2.cvtColor(curr_ref_img, cv2.COLOR_RGB2LAB)
            gen_lab = cv2.cvtColor(curr_gen_img, cv2.COLOR_RGB2LAB)

            # 4. 统计量计算 (使用中位数完美规避图案干扰)
            if valid_pixels_bool is not None:
                ref_valid = ref_lab[valid_pixels_bool]
                gen_valid = gen_lab[valid_pixels_bool]
            else:
                ref_valid = ref_lab.reshape(-1, 3)
                gen_valid = gen_lab.reshape(-1, 3)

            if ref_valid.size == 0 or gen_valid.size == 0:
                ref_valid = ref_lab.reshape(-1, 3)
                gen_valid = gen_lab.reshape(-1, 3)

            r_mean = np.median(ref_valid, axis=0)
            g_mean = np.median(gen_valid, axis=0)
            
            r_std  = np.std(ref_valid, axis=0) + 1e-6 
            g_std  = np.std(gen_valid, axis=0) + 1e-6

            del ref_valid
            del gen_valid
            del ref_lab

            # 5. 应用颜色迁移 (核心修复区)
            l_channel = gen_lab[:, :, 0]
            ab_channels = gen_lab[:, :, 1:]

            l_shift = float(r_mean[0] - g_mean[0])
            ab_shift = r_mean[1:] - g_mean[1:]
            
            # 限制全局最大偏移幅度
            ab_shift = np.clip(ab_shift, -15.0, 15.0)
            l_shift = np.clip(l_shift, -25.0, 25.0)
            ab_shift_vec = ab_shift.reshape(1, 1, 2)

            # --- 修复点 A：区域隔离隔离 (保护背景) ---
            if curr_mask is not None:
                weight_target = 1.0 - curr_mask  # 目标区域权重为1，背景为0
            else:
                weight_target = np.ones_like(l_channel)

            # --- 修复点 B：色域容差保护 (Gamut Capacity) ---
            # 物理法则：极亮(>80)或极暗(<20)的像素容纳不下高饱和度，强行注入会导致RGB严重溢出（即噪点根源）
            capacity = np.ones_like(l_channel)
            # 让高光区域渐渐失去强上色能力
            capacity[l_channel > 80] = (100.0 - l_channel[l_channel > 80]) / 20.0
            # 让阴影死黑区域也不吸收过量颜色
            capacity[l_channel < 20] = l_channel[l_channel < 20] / 20.0
            capacity = np.clip(capacity, 0.0, 1.0)

            # 综合颜色权重：遮罩权重 * 色域容差权重
            final_ab_weight = weight_target * capacity
            final_ab_weight = final_ab_weight[..., np.newaxis] # 匹配通道维度
            
            # 明度不需要色域限制，只需遮罩限制
            final_l_weight = weight_target

            if method == "reinhard_lab":
                scale = r_std[1:] / g_std[1:]
                scale = np.clip(scale, 0.5, 1.5).reshape(1, 1, 2)
                
                # 计算目标颜色，并用 final_ab_weight 平滑过渡
                target_ab = (ab_channels - g_mean[1:].reshape(1,1,2)) * scale + (g_mean[1:].reshape(1,1,2) + ab_shift_vec)
                ab_channels = ab_channels * (1.0 - final_ab_weight) + target_ab * final_ab_weight
                
                # 同步调整亮度
                l_channel += (l_shift * 0.7) * final_l_weight
                
            elif method == "mkl_neutral":
                # 只在目标衣服区域 + 根据高光阴影宽容度 智能加上偏移量
                ab_channels += ab_shift_vec * final_ab_weight
                l_channel += (l_shift * 0.7) * final_l_weight

            gen_lab[:, :, 0] = l_channel
            gen_lab[:, :, 1:] = ab_channels

            # 6. 转回 RGB
            res_rgb = cv2.cvtColor(gen_lab, cv2.COLOR_LAB2RGB)
            res_rgb = np.clip(res_rgb, 0.0, 1.0)
            
            del gen_lab

            # 7. 混合 (Blend)
            if blend_factor < 1.0:
                res_rgb = res_rgb * blend_factor + curr_gen_img * (1 - blend_factor)
                res_rgb = np.clip(res_rgb, 0.0, 1.0)

            img_tensor = torch.from_numpy(res_rgb)
            out_tensors.append(img_tensor)
            
            del res_rgb
            del curr_gen_img

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