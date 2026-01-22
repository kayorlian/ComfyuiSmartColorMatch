import torch
import numpy as np
import cv2
import gc # 引入垃圾回收模块

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
                "ignore_mask": ("MASK",), # 遮罩
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "Image/Color"

    # 添加推理模式装饰器，防止梯度计算占用内存
    @torch.inference_mode()
    def match_color(self, image_ref, image_gen, method, blend_factor, ignore_mask=None):
        try:
            # 1. 预处理：获取目标尺寸
            # 我们只处理 Batch 中的第一张图
            target_h, target_w = image_gen.shape[1], image_gen.shape[2]
            
            # --- 优化点 1: 在 Tensor 阶段就进行 Resize，避免大图转 Numpy 爆内存 ---
            # image_ref 形状是 [B, H, W, C]，PyTorch interpolate 需要 [B, C, H, W]
            ref_tensor = image_ref[0].unsqueeze(0).permute(0, 3, 1, 2) # [1, C, H, W]
            
            # 如果参考图尺寸远大于目标图，先在 GPU/Tensor 层面缩小，节省大量 CPU 内存
            if ref_tensor.shape[2] != target_h or ref_tensor.shape[3] != target_w:
                ref_tensor = torch.nn.functional.interpolate(
                    ref_tensor, size=(target_h, target_w), mode="area"
                )
            
            # 转回 [H, W, C] 并转 Numpy
            ref_tensor = ref_tensor.permute(0, 2, 3, 1).squeeze(0)
            ref_np = (ref_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # 处理生成图
            gen_np = (image_gen[0].cpu().numpy() * 255).astype(np.uint8)

            # 2. 处理 Mask
            valid_pixels_bool = None
            if ignore_mask is not None:
                mask_np = ignore_mask.cpu().numpy()
                if mask_np.ndim == 3:
                    mask_np = mask_np[0] 
                
                # 确保 Mask 尺寸匹配
                if mask_np.shape != (target_h, target_w):
                    mask_np = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                
                # < 0.5 意味着选中背景(黑色区域)参与计算
                valid_pixels_bool = mask_np < 0.5
            else:
                valid_pixels_bool = np.ones((target_h, target_w), dtype=bool)

            # 3. 转换到 LAB 空间
            ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB).astype(np.float32)

            # 4. 提取有效像素计算统计量
            ref_valid = ref_lab[valid_pixels_bool]
            gen_valid = gen_lab[valid_pixels_bool]

            if ref_valid.size == 0 or gen_valid.size == 0:
                print("SmartColorMatch Warning: Mask covers entire image or is empty, using global stats.")
                ref_valid = ref_lab.reshape(-1, 3)
                gen_valid = gen_lab.reshape(-1, 3)

            r_mean = np.mean(ref_valid, axis=0)
            r_std  = np.std(ref_valid, axis=0) + 1e-5
            
            g_mean = np.mean(gen_valid, axis=0)
            g_std  = np.std(gen_valid, axis=0) + 1e-5

            # 5. 应用颜色迁移
            res_lab = gen_lab.copy()

            if method == "reinhard_lab":
                scale = r_std[1:] / g_std[1:]
                centered = gen_lab[:, :, 1:] - g_mean[1:]
                res_lab[:, :, 1:] = centered * scale + r_mean[1:]

            elif method == "mkl_neutral":
                res_lab[:, :, 1:] = (gen_lab[:, :, 1:] - g_mean[1:]) + r_mean[1:]

            # 6. 限制范围并转回 RGB
            res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
            res_rgb = cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB)

            # 7. 混合 (Blend)
            final_rgb = (res_rgb.astype(np.float32) * blend_factor + gen_np.astype(np.float32) * (1 - blend_factor))
            final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)

            # 8. 转回 Tensor
            img_out = torch.from_numpy(final_rgb).float() / 255.0
            img_out = img_out.unsqueeze(0) 

            # --- 优化点 2: 显式清理大变量 ---
            del ref_np, gen_np, ref_lab, gen_lab, res_lab, res_rgb, final_rgb, ref_valid, gen_valid
            if ignore_mask is not None:
                del mask_np
            
            return (img_out,)

        finally:
            # --- 优化点 3: 强制垃圾回收 ---
            # 无论程序是否出错，都强制进行垃圾回收
            gc.collect() 
            torch.cuda.empty_cache() # 如果有 GPU 显存碎片也可以尝试清理（可选）

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SmartColorMatch": SmartColorMatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatch": "Smart Color Match (Masked)"
}
