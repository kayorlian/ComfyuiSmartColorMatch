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
                "invert_mask": (["false", "true"], {"default": "false"}),
                # 🌟 新增：透视调试模式。如果依然颜色不对，打开它，你能亲眼看到算法提取了什么部分！
                "debug_view": (["false", "true"], {"default": "false"}),
            },
            "optional": {
                "ignore_mask": ("MASK",), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "Image/Color"

    # 🛡️ 防弹级图像提取器
    def _extract_image(self, tensor):
        t = tensor.cpu().float()
        if t.ndim == 4: t = t[0] # 取第一帧 [H, W, C] 或 [C, H, W]
        # 修复 PyTorch 2.9+ 某些节点输出通道前置 [C, H, W] 的致命 Bug
        if t.shape[0] in [1, 3, 4] and t.shape[-1] > 4: 
            t = t.permute(1, 2, 0)
        # 强制整理内存布局，防止 Numpy 读出乱码
        t = t.contiguous().numpy()
        # 强行剥离透明通道
        if t.shape[-1] > 3: t = t[..., :3]
        return (t * 255.0).clip(0, 255).astype(np.uint8)

    # 🛡️ 防弹级遮罩提取器
    def _extract_mask(self, tensor):
        t = tensor.cpu().float().squeeze() # 暴击降维，把 [1, H, W, 1] 压扁
        if t.ndim == 3: t = t[0]
        return t.contiguous().numpy()

    def match_color(self, image_ref, image_gen, method, blend_factor, invert_mask="false", debug_view="false", ignore_mask=None):
        ref_np = self._extract_image(image_ref)
        gen_np = self._extract_image(image_gen)

        if ref_np.shape != gen_np.shape:
            ref_np = cv2.resize(ref_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_AREA)

        valid_pixels_bool = None
        if ignore_mask is not None:
            mask_np = self._extract_mask(ignore_mask)
            mask_np = cv2.resize(mask_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            if mask_np.max() > 2.0:
                mask_np = mask_np / 255.0
                
            # 🛡️ 核心修复：红裙子防渗漏。强行把衣服遮罩向外膨胀15个像素，确保背景统计时绝对碰不到红裙子的边缘！
            kernel = np.ones((15, 15), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)

            if invert_mask == "true":
                mask_np = 1.0 - mask_np
                
            valid_pixels_bool = mask_np < 0.5
        else:
            valid_pixels_bool = np.ones((gen_np.shape[0], gen_np.shape[1]), dtype=bool)

        # 🌟 调试模式：直接把算法提取的“背景”给你看！
        if debug_view == "true":
            debug_img = gen_np.copy()
            # 把算法【不处理】的衣服部分涂成瞎眼的亮绿色
            debug_img[~valid_pixels_bool] = [0, 255, 0] 
            img_out = torch.from_numpy(debug_img).to(torch.float32) / 255.0
            return (img_out.unsqueeze(0),)

        ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        ref_valid = ref_lab[valid_pixels_bool]
        gen_valid = gen_lab[valid_pixels_bool]

        if len(ref_valid) < 100 or len(gen_valid) < 100:
            print("==== [SmartColorMatch 警告] 遮罩失效，回退全局统计 ====")
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

        img_out = torch.from_numpy(final_rgb).to(torch.float32) / 255.0
        img_out = img_out.unsqueeze(0) 

        return (img_out,)

NODE_CLASS_MAPPINGS = {
    "SmartColorMatch": SmartColorMatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatch": "Smart Color Match (Masked)"
}