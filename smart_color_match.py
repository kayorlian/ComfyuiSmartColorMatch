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
            # 1. 准备数据：保持 Float32 (0.0 - 1.0) 精度，不要转 uint8
            curr_ref_img = image_ref[i % ref_batch_size].cpu().numpy()
            curr_gen_img = image_gen[i].cpu().numpy()

            # 确保尺寸一致 (调整参考图)
            if curr_ref_img.shape[:2] != curr_gen_img.shape[:2]:
                # 使用 LINEAR 缩放以保持平滑，避免产生额外的噪点
                curr_ref_img = cv2.resize(curr_ref_img, (curr_gen_img.shape[1], curr_gen_img.shape[0]), interpolation=cv2.INTER_LINEAR)

            # 2. 处理 Mask
            valid_pixels_bool = None
            if mask_np_batch is not None:
                curr_mask = mask_np_batch[i % mask_np_batch.shape[0]]
                if curr_mask.shape != curr_gen_img.shape[:2]:
                    curr_mask = cv2.resize(curr_mask, (curr_gen_img.shape[1], curr_gen_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                # Mask 通常 1.0 是遮挡，0.0 是内容；或者反之。ComfyUI mask 0是黑，1是白。
                # 假设 Mask 输入是 "ignore_mask" (遮挡部分)，则 < 0.5 (黑色部分) 是我们要处理的区域
                valid_pixels_bool = curr_mask < 0.5

            # 3. 颜色空间转换 RGB -> LAB (Float32)
            # OpenCV float32 LAB 范围: L [0, 100], a [-127, 127], b [-127, 127]
            # 这种精度比 uint8 (0-255) 高得多，能有效避免冷暖色调的色相偏移
            ref_lab = cv2.cvtColor(curr_ref_img, cv2.COLOR_RGB2LAB)
            gen_lab = cv2.cvtColor(curr_gen_img, cv2.COLOR_RGB2LAB)

            # 4. 统计量计算
            if valid_pixels_bool is not None:
                # 展平并筛选
                ref_valid = ref_lab[valid_pixels_bool]
                gen_valid = gen_lab[valid_pixels_bool]
            else:
                ref_valid = ref_lab.reshape(-1, 3)
                gen_valid = gen_lab.reshape(-1, 3)

            # 兜底：防止 Mask 全白导致无像素
            if ref_valid.size == 0 or gen_valid.size == 0:
                ref_valid = ref_lab.reshape(-1, 3)
                gen_valid = gen_lab.reshape(-1, 3)

            # 计算均值和标准差
            r_mean = np.mean(ref_valid, axis=0)
            r_std  = np.std(ref_valid, axis=0) + 1e-6 # 增加极小值防止除零
            g_mean = np.mean(gen_valid, axis=0)
            g_std  = np.std(gen_valid, axis=0) + 1e-6

            del ref_valid
            del gen_valid
            del ref_lab

            # 5. 应用颜色迁移 (操作 a, b 通道)
            l_channel = gen_lab[:, :, 0]
            ab_channels = gen_lab[:, :, 1:]

            if method == "reinhard_lab":
                # 计算缩放系数
                scale = r_std[1:] / g_std[1:]
                
                # [关键修复]：限制饱和度放大的倍率
                # 如果 AI 图很平滑(std小)，原图很噪(std大)，scale 会变得巨大，导致颜色过饱和(青/红偏色)
                # 这里限制 scale 最大为 1.5 倍，最小 0.5 倍，保持相对自然
                scale = np.clip(scale, 0.5, 1.5)
                
                # (X - Mean_src) * Scale + Mean_tgt
                ab_channels -= g_mean[1:]
                ab_channels *= scale
                ab_channels += r_mean[1:]
                
            elif method == "mkl_neutral":
                # 仅迁移均值，不迁移方差 (更稳定，适合光照差异大的情况)
                ab_channels -= g_mean[1:]
                ab_channels += r_mean[1:]

            # 赋回修改后的通道
            gen_lab[:, :, 1:] = ab_channels

            # 6. 转回 RGB
            # Float LAB 转回 RGB 后，值域不一定是 0-1，可能有溢出，需要 Clip
            res_rgb = cv2.cvtColor(gen_lab, cv2.COLOR_LAB2RGB)
            res_rgb = np.clip(res_rgb, 0.0, 1.0)
            
            del gen_lab

            # 7. 混合 (Blend)
            if blend_factor < 1.0:
                res_rgb = res_rgb * blend_factor + curr_gen_img * (1 - blend_factor)
                res_rgb = np.clip(res_rgb, 0.0, 1.0) # 再次确保安全

            # 转 Tensor
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
