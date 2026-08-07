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
        # 1. 提取 Tensor 到 Numpy (全程保持 float32)
        ref_np = image_ref[0].cpu().numpy()
        gen_np = image_gen[0].cpu().numpy()

        # 2. 处理 Mask (不再做二值化一刀切，保留羽化过渡区)
        ref_mask = np.ones((ref_np.shape[0], ref_np.shape[1]), dtype=np.float32)
        gen_mask = np.ones((gen_np.shape[0], gen_np.shape[1]), dtype=np.float32)

        if ignore_mask is not None:
            raw_mask = ignore_mask.cpu().numpy()
            if raw_mask.ndim == 3: 
                raw_mask = raw_mask[0]
            
            # 权重反转：需要的是背景/肤色 (mask 越小，权重越高)
            raw_mask_inv = np.clip(1.0 - raw_mask, 0.0, 1.0)
            
            # 分别对齐尺寸 (空间解耦，只 Resize Mask，不 Resize 图片)
            gen_mask = cv2.resize(raw_mask_inv, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            ref_mask_raw = cv2.resize(raw_mask_inv, (ref_np.shape[1], ref_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # 反弹光物理剥离 (De-contamination)
            if decontaminate_radius > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (decontaminate_radius, decontaminate_radius))
                # 腐蚀背景 Mask（等同于膨胀衣服区），避开服装交界处的环境光污染
                ref_mask = cv2.erode(ref_mask_raw, kernel, iterations=1)
            else:
                ref_mask = ref_mask_raw

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

        # 5. 基于掩膜独立计算目标与源的统计量 (消除空间暴力缩放)
        ref_mean, ref_std = self.calculate_weighted_stats(ref_lab, ref_mask)
        gen_mean, gen_std = self.calculate_weighted_stats(gen_lab_low, gen_mask)

        # 6. 色偏修复运算 (仅在低频层)
        corrected_low = np.copy(gen_lab_low)
        
        # 色相/饱和度偏移量 (A/B 通道)
        delta_a = ref_mean[1] - gen_mean[1]
        delta_b = ref_mean[2] - gen_mean[2]
        # L 通道均值极弱拉扯 (修复全局发灰，同时不破坏立体感)
        delta_l = (ref_mean[0] - gen_mean[0]) * 0.3 

        # 7. 高光与阴影动态保护 (Luminance Roll-off)
        if luma_protection > 0.0:
            # L 通道范围是 0 到 100
            l_channel = gen_lab_low[:, :, 0]
            
            # 构建抛物线权重: 中间调(50)为1，向0和100平滑衰减
            # normalized_l: -1 到 1
            normalized_l = (l_channel - 50.0) / 50.0
            # weight: 1.0 (中间) -> 0.0 (两端)
            roll_off_weight = 1.0 - (np.abs(normalized_l) ** (1.0 / luma_protection))
            roll_off_weight = np.clip(roll_off_weight, 0.0, 1.0)
            roll_off_weight = np.expand_dims(roll_off_weight, axis=-1)
            
            # 应用带保护的偏移
            shifts = np.array([delta_l, delta_a, delta_b])
            corrected_low += shifts * roll_off_weight
        else:
            corrected_low[:, :, 0] += delta_l
            corrected_low[:, :, 1] += delta_a
            corrected_low[:, :, 2] += delta_b

        # 8. 频域重组：加回高频细节层
        final_lab = corrected_low + gen_lab_high

        # 9. 转换回 RGB 并处理溢出边界
        final_lab = np.clip(final_lab, [0, -128, -128], [100, 127, 127])
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