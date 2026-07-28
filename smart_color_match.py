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
                "image_ref": ("IMAGE",),  # 原图
                "image_gen": ("IMAGE",),  # 生成图
                "method": (["mkl_neutral", "reinhard_lab"],), # 算法选择
                "blend_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "ignore_mask": ("MASK",), # 衣服的蒙版（指明哪些地方不需要计算颜色统计）
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "Image/Color"

    def match_color(self, image_ref, image_gen, method, blend_factor, ignore_mask=None):
        # 1. 彻底解决截断问题：保持 float32 格式 (0.0 - 1.0)，不要乘以 255 转为 uint8
        # OpenCV 在处理 float32 的 LAB 转换时，会保留真实的物理数值范围（A/B通道在 -127 到 +127 左右），杜绝强制 0 截断
        ref_np = image_ref[0].cpu().numpy().astype(np.float32)
        gen_np = image_gen[0].cpu().numpy().astype(np.float32)

        # 确保尺寸一致，如果不一致，将 ref 缩放到 gen 的大小
        if ref_np.shape != gen_np.shape:
            ref_np = cv2.resize(ref_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_AREA)

        # 2. 处理 Mask 
        if ignore_mask is not None:
            mask_np = ignore_mask.cpu().numpy()
            if mask_np.ndim == 3: 
                mask_np = mask_np[0] 
            
            # Resize mask to image size
            mask_np = cv2.resize(mask_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 1=Clothes(Ignore), 0=Background(Keep)
            valid_pixels_bool = mask_np < 0.5
        else:
            valid_pixels_bool = np.ones((gen_np.shape[0], gen_np.shape[1]), dtype=bool)

        # 3. 彻底解决统计污染：引入有效区域阈值保护
        total_pixels = gen_np.shape[0] * gen_np.shape[1]
        valid_count = np.sum(valid_pixels_bool)
        
        # 如果有效背景像素不足全图的 5%，极有可能是掩膜漏出的边缘杂色/绿植噪点
        # 此时强制回退到全局统计，防止一张小小的绿图带偏整张大图
        if valid_count / total_pixels < 0.05:
            print("SmartColorMatch: Valid background area too small (<5%). Falling back to global stats to avoid color pollution.")
            valid_pixels_bool = np.ones((gen_np.shape[0], gen_np.shape[1]), dtype=bool)

        # 4. 转换到 LAB 空间 (float32 输入)
        ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB)
        gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB)

        # 提取有效像素
        ref_valid = ref_lab[valid_pixels_bool]
        gen_valid = gen_lab[valid_pixels_bool]

        # 如果发生极端情况（理论上被上方的 5% 拦截，这里作为最后防线）
        if len(ref_valid) == 0 or len(gen_valid) == 0:
            ref_valid = ref_lab
            gen_valid = gen_lab

        # 计算统计量 (Mean, Std)
        r_mean = np.mean(ref_valid, axis=0)
        r_std  = np.std(ref_valid, axis=0) + 1e-5
        
        g_mean = np.mean(gen_valid, axis=0)
        g_std  = np.std(gen_valid, axis=0) + 1e-5

        res_lab = gen_lab.copy()

        # 5. 核心算法修正
        if method == "reinhard_lab":
            # 彻底解决极值爆炸：对缩放比例进行钳制 (Clamping)
            # 即使原背景纯色导致 std 趋近于 0，缩放系数也被死死卡在 [0.5, 2.0] 倍之间，绝对不会发生乘数爆炸
            std_ratio = r_std / g_std
            std_ratio_clamped = np.clip(std_ratio, 0.5, 2.0)

            for i in [1, 2]: # 1=A, 2=B
                res_lab[:,:,i] = (gen_lab[:,:,i] - g_mean[i]) * std_ratio_clamped[i] + r_mean[i]
                
        elif method == "mkl_neutral":
            for i in [1, 2]:
                res_lab[:,:,i] = (gen_lab[:,:,i] - g_mean[i]) + r_mean[i]

        # 6. 彻底解决色域溢出：放弃对 LAB 直接进行 np.clip
        # 先利用 cv2 将浮点数的 LAB 转回 RGB，让算法自然映射
        res_rgb = cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB)

        # 此时再对物理 RGB 值进行 0.0 到 1.0 的安全裁切，保证不产生突兀的灰绿断层
        res_rgb = np.clip(res_rgb, 0.0, 1.0)

        # 7. Blend 混合 (直接使用 float 进行运算)
        final_rgb = res_rgb * blend_factor + gen_np * (1.0 - blend_factor)

        # 8. 转回 ComfyUI 所需的 Tensor 格式 [B, H, W, C]
        img_out = torch.from_numpy(final_rgb).unsqueeze(0)

        return (img_out,)

NODE_CLASS_MAPPINGS = {
    "SmartColorMatch": SmartColorMatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatch": "Smart Color Match (Fixed)"
}