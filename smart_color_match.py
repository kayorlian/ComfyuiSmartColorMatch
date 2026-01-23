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
                "image_ref": ("IMAGE",),  # 原图 (参考图) [B, H, W, C]
                "image_gen": ("IMAGE",),  # 生成图 (目标图) [B, H, W, C]
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
        # 获取 Batch 大小
        batch_size = image_gen.shape[0]
        ref_batch_size = image_ref.shape[0]
        
        out_tensors = []

        # 预处理 Mask (如果存在)
        mask_np_batch = None
        if ignore_mask is not None:
            # Mask 通常是 [B, H, W] 或 [1, H, W]
            mask_np_batch = ignore_mask.cpu().numpy()
            if mask_np_batch.ndim == 2:
                mask_np_batch = mask_np_batch[np.newaxis, ...] # 补齐 Batch 维度

        for i in range(batch_size):
            # 1. 准备当前帧数据
            # 如果参考图少于生成图，循环使用参考图
            curr_ref_img = image_ref[i % ref_batch_size] 
            curr_gen_img = image_gen[i]

            # 转换为 Numpy (H, W, 3) uint8
            ref_np = (curr_ref_img.cpu().numpy() * 255).astype(np.uint8)
            gen_np = (curr_gen_img.cpu().numpy() * 255).astype(np.uint8)

            # 确保尺寸一致 (调整参考图以匹配生成图)
            if ref_np.shape[:2] != gen_np.shape[:2]:
                ref_np = cv2.resize(ref_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_AREA)

            # 2. 处理 Mask
            valid_pixels_bool = None
            if mask_np_batch is not None:
                # 处理 Mask 的 Batch 广播 (如果 Mask 只有1张，应用于所有图片)
                curr_mask = mask_np_batch[i % mask_np_batch.shape[0]]
                
                # 调整 Mask 尺寸
                if curr_mask.shape != gen_np.shape[:2]:
                    curr_mask = cv2.resize(curr_mask, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # < 0.5 意味着选中黑色区域参与计算
                valid_pixels_bool = curr_mask < 0.5
            else:
                # 即使没有 Mask，也尽量不要创建全为 True 的巨型 bool 数组，直接用 None 标记
                valid_pixels_bool = None

            # 3. 颜色空间转换 RGB -> LAB
            ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB).astype(np.float32)

            # 4. 统计量计算
            # 提取有效像素
            if valid_pixels_bool is not None:
                ref_valid = ref_lab[valid_pixels_bool]
                gen_valid = gen_lab[valid_pixels_bool]
            else:
                # 使用 reshape 避免 copy，减少内存占用
                ref_valid = ref_lab.reshape(-1, 3)
                gen_valid = gen_lab.reshape(-1, 3)

            # 兜底：如果 Mask 导致无像素
            if ref_valid.size == 0 or gen_valid.size == 0:
                ref_valid = ref_lab.reshape(-1, 3)
                gen_valid = gen_lab.reshape(-1, 3)

            # 计算均值和标准差
            r_mean = np.mean(ref_valid, axis=0)
            r_std  = np.std(ref_valid, axis=0) + 1e-5
            g_mean = np.mean(gen_valid, axis=0)
            g_std  = np.std(gen_valid, axis=0) + 1e-5
            
            # 释放不再需要的统计样本内存
            del ref_valid
            del gen_valid
            # ref_lab 也不再需要了，只需要它的统计数据
            del ref_lab

            # 5. 应用颜色迁移 (直接修改 gen_lab，避免创建 copy)
            # LAB: L=0, A=1, B=2. 我们只修改 A 和 B
            
            l_channel = gen_lab[:, :, 0] # 引用 L 通道
            ab_channels = gen_lab[:, :, 1:] # 引用 AB 通道
            
            if method == "reinhard_lab":
                scale = r_std[1:] / g_std[1:]
                # (X - Mean_src) * Scale + Mean_tgt
                # 尽量使用原地操作符 -=, *=, += 减少临时内存分配
                ab_channels -= g_mean[1:] 
                ab_channels *= scale
                ab_channels += r_mean[1:]
                
            elif method == "mkl_neutral":
                # (X - Mean_src) + Mean_tgt
                ab_channels -= g_mean[1:]
                ab_channels += r_mean[1:]

            # 将修改后的 AB 通道赋回 (其实上面的切片引用已经是 View 了，但为了保险)
            gen_lab[:, :, 1:] = ab_channels

            # 6. 转回 RGB
            # Clip 并转 uint8
            np.clip(gen_lab, 0, 255, out=gen_lab) # 原地 clip
            gen_lab_uint8 = gen_lab.astype(np.uint8)
            del gen_lab # 释放 float32 大数组

            res_rgb = cv2.cvtColor(gen_lab_uint8, cv2.COLOR_LAB2RGB)
            del gen_lab_uint8

            # 7. 混合 (Blend)
            if blend_factor < 1.0:
                # 转换为 float32 进行混合
                res_rgb = res_rgb.astype(np.float32) * blend_factor + gen_np.astype(np.float32) * (1 - blend_factor)
                np.clip(res_rgb, 0, 255, out=res_rgb)
                res_rgb = res_rgb.astype(np.uint8)
            
            del gen_np # 释放原图 Numpy

            # 转 Tensor 并立即归一化，释放 Numpy 内存
            img_tensor = torch.from_numpy(res_rgb).float() / 255.0
            out_tensors.append(img_tensor)
            
            del res_rgb
            
            # 手动触发 GC 并不是必须的，但在处理极大图片 Batch 时可以防止峰值过高
            # gc.collect() 

        # 拼接 Batch: [B, H, W, C]
        if len(out_tensors) > 0:
            final_output = torch.stack(out_tensors, dim=0)
        else:
            final_output = image_gen # 兜底

        return (final_output,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SmartColorMatch": SmartColorMatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatch": "Smart Color Match (Masked)"
}
