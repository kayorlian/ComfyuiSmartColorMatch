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
                "image_ref": ("IMAGE",),  
                "image_gen": ("IMAGE",),  
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
        # 1. 终极修复：强制转换为标准的 float32，彻底杜绝 PyTorch 2.9+ 中的 bfloat16/float16 导致的 numpy 乱码崩溃
        ref_tensor = image_ref.to(torch.float32).cpu()
        gen_tensor = image_gen.to(torch.float32).cpu()
        
        # 确保只取第一帧 (降维提取)
        if ref_tensor.ndim == 4: ref_tensor = ref_tensor[0]
        if gen_tensor.ndim == 4: gen_tensor = gen_tensor[0]

        # 安全转为 Numpy uint8
        ref_np = (ref_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
        gen_np = (gen_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # 终极修复 2：如果新版 ComfyUI 传来了带透明度 (RGBA) 的 4 通道图，强行丢弃 Alpha 通道，只留 RGB
        if ref_np.shape[-1] > 3: ref_np = ref_np[..., :3]
        if gen_np.shape[-1] > 3: gen_np = gen_np[..., :3]

        if ref_np.shape != gen_np.shape:
            ref_np = cv2.resize(ref_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_AREA)

        # 2. 处理 Mask
        valid_mask = None
        if ignore_mask is not None:
            # 同样强制 float32，保住 Mask 的数据纯洁性
            mask_tensor = ignore_mask.to(torch.float32).cpu()
            mask_np = mask_tensor.numpy()
            
            # 终极修复 3：不论上游传来的是 [B, H, W], [H, W] 还是 [B, 1, H, W]，暴击降维只拿 2D 数组
            while mask_np.ndim > 2:
                mask_np = mask_np[0]
                
            mask_np = cv2.resize(mask_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 兼容各种变态的 Mask 数值范围（0-255 或 0.0-1.0）
            if mask_np.max() > 2.0:
                mask_np = mask_np / 255.0
            
            valid_pixels_bool = mask_np < 0.5
        else:
            valid_pixels_bool = np.ones((gen_np.shape[0], gen_np.shape[1]), dtype=bool)

        # 3. 转换到 LAB 空间
        ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        # 4. 核心算法提取
        ref_valid = ref_lab[valid_pixels_bool]
        gen_valid = gen_lab[valid_pixels_bool]

        # 验证提取是否成功，如果提取的有效像素太少，说明遮罩失效，给出明显警告
        if len(ref_valid) < 100 or len(gen_valid) < 100:
            print("===================================================")
            print("[SmartColorMatch WARNING] MASK 读取失效或覆盖了全图！")
            print("正在回退为全局统计，这大概率会导致画面的肤色严重偏蓝/偏色！")
            print("===================================================")
            ref_valid = ref_lab.reshape(-1, 3)
            gen_valid = gen_lab.reshape(-1, 3)
        else:
            ref_valid = ref_valid.reshape(-1, 3)
            gen_valid = gen_valid.reshape(-1, 3)

        r_mean = np.mean(ref_valid, axis=0)
        r_std  = np.std(ref_valid, axis=0) + 1e-5
        
        g_mean = np.mean(gen_valid, axis=0)
        g_std  = np.std(gen_valid, axis=0) + 1e-5
        
        res_lab = gen_lab.copy()

        if method == "reinhard_lab":
            for i in [1, 2]: 
                res_lab[:,:,i] = (gen_lab[:,:,i] - g_mean[i]) * (r_std[i] / g_std[i]) + r_mean[i]
        elif method == "mkl_neutral":
            for i in [1, 2]:
                res_lab[:,:,i] = (gen_lab[:,:,i] - g_mean[i]) + r_mean[i]

        res_lab = np.clip(res_lab, 0, 255)
        res_rgb = cv2.cvtColor(res_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        final_rgb = (res_rgb * blend_factor + gen_np * (1 - blend_factor)).astype(np.uint8)

        # 输出前再次转回标准的 PyTorch float32
        img_out = torch.from_numpy(final_rgb).to(torch.float32) / 255.0
        img_out = img_out.unsqueeze(0) 

        return (img_out,)

NODE_CLASS_MAPPINGS = {
    "SmartColorMatch": SmartColorMatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatch": "Smart Color Match (Masked)"
}