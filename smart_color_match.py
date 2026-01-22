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
                "ignore_mask": ("MASK",), # 遮罩：默认白色区域被忽略，计算黑色区域
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "Image/Color"

    def match_color(self, image_ref, image_gen, method, blend_factor, ignore_mask=None):
        # 1. 转换图像: Tensor [B, H, W, C] -> Numpy [H, W, C] (0-255)
        # ComfyUI 图片默认是 RGB 格式
        if image_ref.shape[0] > 1 or image_gen.shape[0] > 1:
            print("SmartColorMatch: Warning - Only processing the first image in the batch.")

        ref_np = (image_ref[0].cpu().numpy() * 255).astype(np.uint8)
        gen_np = (image_gen[0].cpu().numpy() * 255).astype(np.uint8)

        # 确保尺寸一致
        if ref_np.shape != gen_np.shape:
            ref_np = cv2.resize(ref_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_AREA)

        # 2. 处理 Mask
        # 这里的逻辑是：Mask 为 1 (白色) 的地方被忽略 (Ignore)，Mask 为 0 (黑色) 的地方参与计算 (Keep)。
        # 如果你传入的是“人物Mask”，但想匹配人物颜色，应该反转Mask，或者不传 ignore_mask。
        valid_pixels_bool = None
        if ignore_mask is not None:
            mask_np = ignore_mask.cpu().numpy()
            if mask_np.ndim == 3:
                mask_np = mask_np[0] # 取 Batch 第一张
            
            # 确保 Mask 尺寸匹配生成图
            if mask_np.shape != (gen_np.shape[0], gen_np.shape[1]):
                mask_np = cv2.resize(mask_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # < 0.5 意味着选中背景(黑色区域)。如果Mask是人物，这里选中的就是非人物区域。
            # 如果背景是蓝色，人物就会被染成蓝色。
            valid_pixels_bool = mask_np < 0.5
        else:
            # 没有 Mask 则全图计算
            valid_pixels_bool = np.ones((gen_np.shape[0], gen_np.shape[1]), dtype=bool)

        # 3. 转换到 LAB 空间 (保持 RGB -> LAB)
        # 必须确保输入是 uint8 且转换码正确，否则颜色会乱
        ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        # 4. 提取有效像素计算统计量
        ref_valid = ref_lab[valid_pixels_bool]
        gen_valid = gen_lab[valid_pixels_bool]

        # 兜底：如果 Mask 覆盖全图导致无像素，回退到全局统计
        if ref_valid.size == 0 or gen_valid.size == 0:
            print("SmartColorMatch Warning: Mask covers entire image or is empty, using global stats.")
            ref_valid = ref_lab.reshape(-1, 3)
            gen_valid = gen_lab.reshape(-1, 3)

        # 计算均值和标准差 (axis=0 对列求均值，结果 shape=(3,))
        r_mean = np.mean(ref_valid, axis=0)
        r_std  = np.std(ref_valid, axis=0) + 1e-5
        
        g_mean = np.mean(gen_valid, axis=0)
        g_std  = np.std(gen_valid, axis=0) + 1e-5

        # 5. 应用颜色迁移
        res_lab = gen_lab.copy()

        # === 核心修复：向量化运算代替循环，解决 Broadcasting Error ===
        # LAB 通道：0=L (亮度), 1=A (红绿), 2=B (蓝黄)
        # 我们只修改 A 和 B (即索引 1:)，保留 L
        
        # gen_lab[:, :, 1:] 形状是 (H, W, 2)
        # g_mean[1:] 形状是 (2,)
        # NumPy 会自动对齐最后一个维度，不会再报错
        
        if method == "reinhard_lab":
            # (X - Mean_src) * (Std_tgt / Std_src) + Mean_tgt
            scale = r_std[1:] / g_std[1:]
            
            # 1. 去均值 (Center)
            centered = gen_lab[:, :, 1:] - g_mean[1:]
            
            # 2. 缩放标准差并加回目标均值
            res_lab[:, :, 1:] = centered * scale + r_mean[1:]

        elif method == "mkl_neutral":
            # 仅对齐均值 (White Balance): (X - Mean_src) + Mean_tgt
            res_lab[:, :, 1:] = (gen_lab[:, :, 1:] - g_mean[1:]) + r_mean[1:]

        # 6. 限制范围并转回 RGB
        # 注意：先 clip 到 0-255，再转 uint8，最后转 RGB
        res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
        res_rgb = cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB)

        # 7. 混合 (Blend)
        final_rgb = (res_rgb.astype(np.float32) * blend_factor + gen_np.astype(np.float32) * (1 - blend_factor))
        final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)

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
