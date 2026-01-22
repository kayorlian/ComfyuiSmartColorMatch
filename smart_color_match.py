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

    # 使用 inference_mode 避免 PyTorch 记录梯度，进一步节省显存/内存
    @torch.inference_mode()
    def match_color(self, image_ref, image_gen, method, blend_factor, ignore_mask=None):
        # 初始化变量
        ref_np = gen_np = ref_lab = gen_lab = res_lab = res_rgb = final_rgb = None
        
        try:
            target_h, target_w = image_gen.shape[1], image_gen.shape[2]

            # 1. Tensor 预处理 (保持你的优秀逻辑)
            ref_tensor = image_ref[0].unsqueeze(0).permute(0, 3, 1, 2)
            if ref_tensor.shape[2] > target_h or ref_tensor.shape[3] > target_w:
                ref_tensor = torch.nn.functional.interpolate(
                    ref_tensor, size=(target_h, target_w), mode="area"
                )
            ref_tensor = ref_tensor.permute(0, 2, 3, 1).squeeze(0)
            
            # 转 Numpy
            ref_np = (ref_tensor.cpu().numpy() * 255).astype(np.uint8)
            gen_np = (image_gen[0].cpu().numpy() * 255).astype(np.uint8)

            # 2. Mask 处理 (保持不变)
            valid_pixels_bool = None
            if ignore_mask is not None:
                mask_np = ignore_mask.cpu().numpy()
                if mask_np.ndim == 3: mask_np = mask_np[0]
                if mask_np.shape != (target_h, target_w):
                    mask_np = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                valid_pixels_bool = mask_np < 0.5
            else:
                valid_pixels_bool = np.ones((target_h, target_w), dtype=bool)

            # 3. 转 LAB
            ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB).astype(np.float32)

            # 4. 统计量计算 (保持不变)
            ref_valid = ref_lab[valid_pixels_bool]
            gen_valid = gen_lab[valid_pixels_bool]
            
            if ref_valid.size == 0 or gen_valid.size == 0:
                ref_valid = ref_lab.reshape(-1, 3)
                gen_valid = gen_lab.reshape(-1, 3)

            r_mean = np.mean(ref_valid, axis=0)
            r_std  = np.std(ref_valid, axis=0) + 1e-5
            g_mean = np.mean(gen_valid, axis=0)
            g_std  = np.std(gen_valid, axis=0) + 1e-5

            # 5. 颜色迁移
            res_lab = gen_lab.copy()
            if method == "reinhard_lab":
                scale = r_std[1:] / g_std[1:]
                centered = gen_lab[:, :, 1:] - g_mean[1:]
                res_lab[:, :, 1:] = centered * scale + r_mean[1:]
            elif method == "mkl_neutral":
                res_lab[:, :, 1:] = (gen_lab[:, :, 1:] - g_mean[1:]) + r_mean[1:]

            # 6. 转回 RGB
            res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
            res_rgb = cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB)

            # 7. 混合 (优化点：使用 cv2.addWeighted 节省内存)
            # 公式: src1 * alpha + src2 * beta + gamma
            # blend_factor 越大，res_rgb 占比越高
            if blend_factor >= 1.0:
                final_rgb = res_rgb
            elif blend_factor <= 0.0:
                final_rgb = gen_np
            else:
                # 这一步比纯 Numpy 数学运算更省内存且极快
                final_rgb = cv2.addWeighted(res_rgb, blend_factor, gen_np, 1.0 - blend_factor, 0)

            # 8. 输出
            img_out = torch.from_numpy(final_rgb).float() / 255.0
            img_out = img_out.unsqueeze(0)

            return (img_out,)

        except Exception as e:
            print(f"SmartColorMatch Error: {e}")
            return (image_gen,)

        finally:
            # 清理引用
            del ref_np, gen_np, ref_lab, gen_lab, res_lab, res_rgb, final_rgb
            # 强制 GC
            gc.collect()
            # 移除了 torch.cuda.empty_cache() 以避免降速

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SmartColorMatch": SmartColorMatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatch": "Smart Color Match (Masked)"
}
