import torch
import numpy as np
import cv2

class SmartColorMatchAdvanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_ref": ("IMAGE",),  # 原始参考图
                "image_gen": ("IMAGE",),  # 4K 生成图 (带偏色)
                "blend_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # 新增核心控制参数
                "freq_separation_radius": ("INT", {"default": 31, "min": 0, "max": 255, "step": 2}), # 高低频分离半径
                "decontaminate_radius": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1}), # 原图反弹光隔离带宽度
                "luma_protection": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}), # 高光/阴影保护强度
            },
            "optional": {
                "ignore_mask": ("MASK",),  # 1 为服装区(不参与统计)，0 为背景肤色区(参与统计)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "Image/Color Advanced"

    def calculate_weighted_stats(self, image_lab, mask):
        """计算带权重的均值和方差，支持渐变 Mask"""
        mask_expanded = np.expand_dims(mask, axis=-1)
        sum_mask = np.sum(mask_expanded)
        
        if sum_mask < 1e-5:
            return np.zeros(3), np.ones(3)

        # 加权均值
        mean = np.sum(image_lab * mask_expanded, axis=(0, 1)) / sum_mask
        
        # 加权方差
        variance = np.sum(((image_lab - mean) ** 2) * mask_expanded, axis=(0, 1)) / sum_mask
        std = np.sqrt(variance) + 1e-5
        
        return mean, std

    def match_color(self, image_ref, image_gen, blend_factor, freq_separation_radius, decontaminate_radius, luma_protection, ignore_mask=None):
        # 1. 提取 Tensor 到 Numpy (全程保持 float32, ComfyUI 中像素值范围为 0.0 ~ 1.0)
        ref_np = image_ref[0].cpu().numpy()
        gen_np = image_gen[0].cpu().numpy()

        # --- 新增核心逻辑：精准定位纯白像素 (RGB 255, 255, 255 即 1.0, 1.0, 1.0) ---
        # 使用 > 0.99 匹配纯白，以防浮点数精度误差。np.all 确保 R, G, B 三个通道都满足条件。
        ref_is_white = np.all(ref_np > 0.99, axis=-1).astype(np.float32)
        gen_is_white = np.all(gen_np > 0.99, axis=-1).astype(np.float32)
        
        # 反转：1 代表非白（有效像素），0 代表纯白（需过滤的像素）
        ref_valid_pixels = 1.0 - ref_is_white
        gen_valid_pixels = 1.0 - gen_is_white
        # -----------------------------------------------------------------------

        # 2. 处理 Mask (不再做二值化一刀切，保留羽化过渡区)
        ref_mask = np.copy(ref_valid_pixels)
        gen_mask = np.copy(gen_valid_pixels)

        if ignore_mask is not None:
            raw_mask = ignore_mask.cpu().numpy()
            if raw_mask.ndim == 3: 
                raw_mask = raw_mask[0]
            
            # 权重反转：需要的是背景/肤色 (mask 越小，权重越高)
            raw_mask_inv = np.clip(1.0 - raw_mask, 0.0, 1.0)
            
            # 分别对齐尺寸 (空间解耦，只 Resize Mask，不 Resize 图片)
            gen_mask_base = cv2.resize(raw_mask_inv, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            ref_mask_base = cv2.resize(raw_mask_inv, (ref_np.shape[1], ref_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # 反弹光物理剥离 (De-contamination)
            if decontaminate_radius > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (decontaminate_radius, decontaminate_radius))
                # 腐蚀背景 Mask（等同于膨胀衣服区），避开服装交界处的环境光污染
                ref_mask_base = cv2.erode(ref_mask_base, kernel, iterations=1)
            
            # 将用户输入的遮罩与“非白像素”遮罩正片叠底（相乘）
            ref_mask = ref_mask_base * ref_valid_pixels
            gen_mask = gen_mask_base * gen_valid_pixels

        # 3. 转换到 LAB 色彩空间 (Float32 避免断层)
        ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        # 4. 高低频分离 (Frequency Separation)
        if freq_separation_radius > 0:
            k_size = freq_separation_radius if freq_separation_radius % 2 == 1 else freq_separation_radius + 1
            gen_lab_low = cv2.GaussianBlur(gen_lab, (k_size, k_size), 0)
            gen_lab_high = gen_lab - gen_lab_low
        else:
            gen_lab_low = gen_lab
            gen_lab_high = np.zeros_like(gen_lab)

        # 5. 基于掩膜独立计算目标与源的统计量 (消除空间暴力缩放，并自动忽略纯白)
        ref_mean, ref_std = self.calculate_weighted_stats(ref_lab, ref_mask)
        gen_mean, gen_std = self.calculate_weighted_stats(gen_lab_low, gen_mask)

        # 6. 色偏修复运算 (引入标准差缩放，实现完整的 Reinhard 色彩迁移)
        corrected_low = np.copy(gen_lab_low)
        
        # 计算标准差缩放比例 (限制缩放阈值以防极端噪点过度放大)
        ratio_l = np.clip(ref_std[0] / gen_std[0], 0.5, 2.0)
        ratio_a = np.clip(ref_std[1] / gen_std[1], 0.5, 2.0)
        ratio_b = np.clip(ref_std[2] / gen_std[2], 0.5, 2.0)

        # 7. 高光与阴影动态保护 (Luminance Roll-off)
        if luma_protection > 0.0:
            l_channel = gen_lab_low[:, :, 0]
            normalized_l = (l_channel - 50.0) / 50.0
            roll_off_weight = 1.0 - (np.abs(normalized_l) ** (1.0 / luma_protection))
            roll_off_weight = np.clip(roll_off_weight, 0.0, 1.0)
            roll_off_weight = np.expand_dims(roll_off_weight, axis=-1)
            
            # 使用均值平移和标准差缩放，并结合保护权重
            # L 通道极弱拉扯 (0.3 权重保持画面原有光影立体感)
            corrected_low[:, :, 0] = (gen_lab_low[:, :, 0] - gen_mean[0]) * (1.0 + (ratio_l - 1.0) * roll_off_weight[:, :, 0]) + gen_mean[0] + (ref_mean[0] - gen_mean[0]) * 0.3 * roll_off_weight[:, :, 0]
            
            # A, B 通道执行完整映射
            corrected_low[:, :, 1] = (gen_lab_low[:, :, 1] - gen_mean[1]) * (1.0 + (ratio_a - 1.0) * roll_off_weight[:, :, 0]) + gen_mean[1] + (ref_mean[1] - gen_mean[1]) * roll_off_weight[:, :, 0]
            corrected_low[:, :, 2] = (gen_lab_low[:, :, 2] - gen_mean[2]) * (1.0 + (ratio_b - 1.0) * roll_off_weight[:, :, 0]) + gen_mean[2] + (ref_mean[2] - gen_mean[2]) * roll_off_weight[:, :, 0]
        else:
            # 无保护模式：标准的 Reinhard 映射
            corrected_low[:, :, 0] = (gen_lab_low[:, :, 0] - gen_mean[0]) * ratio_l + gen_mean[0] + (ref_mean[0] - gen_mean[0]) * 0.3
            corrected_low[:, :, 1] = (gen_lab_low[:, :, 1] - gen_mean[1]) * ratio_a + ref_mean[1]
            corrected_low[:, :, 2] = (gen_lab_low[:, :, 2] - gen_mean[2]) * ratio_b + ref_mean[2]

        # 8. 频域重组：对高频层同步进行色度方差缩放，彻底消除残留在纹理中的紫偏噪点
        gen_lab_high[:, :, 1] *= ratio_a
        gen_lab_high[:, :, 2] *= ratio_b
        final_lab = corrected_low + gen_lab_high

        # 9. 转换回 RGB 并处理溢出边界
        final_lab = np.clip(final_lab, [0, -128, -128], [100, 127, 127]).astype(np.float32)
        res_rgb = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)
        res_rgb = np.clip(res_rgb, 0.0, 1.0)

        # 10. 全局混合控制 (Blend)
        final_rgb = res_rgb * blend_factor + gen_np * (1.0 - blend_factor)

        # 返回 Tensor [1, H, W, C]
        img_out = torch.from_numpy(final_rgb).unsqueeze(0).float()

        return (img_out,)

NODE_CLASS_MAPPINGS = {
    "SmartColorMatchAdvanced": SmartColorMatchAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatchAdvanced": "Smart Color Match (Ultimate VTON)"
}