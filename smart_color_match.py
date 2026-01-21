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
                "image_ref": ("IMAGE",),  # 原图 (参考图)
                "image_gen": ("IMAGE",),  # 生成图 (目标图)
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
        # 1. ComfyUI 的图片是 Tensor [B, H, W, C] 范围 0-1，转为 Numpy [H, W, C] 范围 0-255
        # 这里默认处理 Batch 中的第一张图片，如果需要处理 Batch，需要外层循环
        if image_ref.shape[0] > 1 or image_gen.shape[0] > 1:
            print("SmartColorMatch: Warning - Only processing the first image in the batch.")

        ref_np = (image_ref[0].cpu().numpy() * 255).astype(np.uint8)
        gen_np = (image_gen[0].cpu().numpy() * 255).astype(np.uint8)

        # 确保尺寸一致，如果不一致，将 ref 缩放到 gen 的大小
        if ref_np.shape != gen_np.shape:
            ref_np = cv2.resize(ref_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_AREA)

        # 2. 处理 Mask
        # 如果传入了 mask，mask 为 1 的地方是衣服（变化的），我们要忽略它，只取 mask 为 0 的地方（背景）
        # ComfyUI Mask 通常是 [B, H, W] 或 [H, W]
        valid_pixels_bool = None
        if ignore_mask is not None:
            mask_np = ignore_mask.cpu().numpy()
            if mask_np.ndim == 3: mask_np = mask_np[0] # 取第一帧
            
            # Resize mask to image size
            mask_np = cv2.resize(mask_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 我们需要的是背景（mask < 0.5 的地方），生成一个布尔索引
            # ignore_mask: 1=Clothes(Ignore), 0=Background(Keep)
            valid_pixels_bool = mask_np < 0.5
        else:
            # 如果没 mask，就用全图计算（回退到普通模式）
            valid_pixels_bool = np.ones((gen_np.shape[0], gen_np.shape[1]), dtype=bool)

        # 3. 转换到 LAB 空间
        ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        # 4. 核心算法：只在 Mask 允许的区域计算 Mean/Std
        # 提取有效像素
        ref_valid = ref_lab[valid_pixels_bool]
        gen_valid = gen_lab[valid_pixels_bool]

        # 如果 mask 覆盖了全图，导致没有有效像素，防报错
        if len(ref_valid) == 0 or len(gen_valid) == 0:
            print("Warning: Mask covers entire image, using global stats.")
            ref_valid = ref_lab.reshape(-1, 3)
            gen_valid = gen_lab.reshape(-1, 3)

        # 计算统计量 (Mean, Std)
        # l, a, b
        r_mean = np.mean(ref_valid, axis=0)
        r_std  = np.std(ref_valid, axis=0) + 1e-5
        
        g_mean = np.mean(gen_valid, axis=0)
        g_std  = np.std(gen_valid, axis=0) + 1e-5

        # 5. 应用颜色迁移到【全图】
        res_lab = gen_lab.copy()

        if method == "reinhard_lab":
            # Reinhard 算法: 只修正 A 和 B 通道 (色相/饱和度)，保留 L 通道 (亮度/光影)
            for i in [1, 2]: # 1=A, 2=B
                res_lab[:,:,i] = (gen_lab[:,:,i] - g_mean[i]) * (r_std[i] / g_std[i]) + r_mean[i]
            # L 通道不动，保留生成图的光影

        elif method == "mkl_neutral":
            # 仅对齐均值（White Balance），不拉伸对比度
            for i in [1, 2]:
                res_lab[:,:,i] = (gen_lab[:,:,i] - g_mean[i]) + r_mean[i]

        # 6. 限制范围并转回 RGB
        res_lab = np.clip(res_lab, 0, 255)
        res_rgb = cv2.cvtColor(res_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        # 7. Blend (与原生成图混合，控制强度)
        final_rgb = (res_rgb * blend_factor + gen_np * (1 - blend_factor)).astype(np.uint8)

        # 转回 Tensor
        img_out = torch.from_numpy(final_rgb).float() / 255.0
        img_out = img_out.unsqueeze(0) # [1, H, W, C]

        return (img_out,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SmartColorMatch": SmartColorMatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatch": "Smart Color Match (Masked)"
}
