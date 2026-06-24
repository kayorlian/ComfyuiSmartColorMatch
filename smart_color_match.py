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
                "fix_bgr": (["true", "false"], {"default": "true"}),
            },
            "optional": {
                "ignore_mask": ("MASK",), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "Image/Color"

    def match_color(self, image_ref, image_gen, method, blend_factor, fix_bgr="true", ignore_mask=None):
        ref_np = (image_ref[0].cpu().numpy() * 255).astype(np.uint8)
        gen_np = (image_gen[0].cpu().numpy() * 255).astype(np.uint8)

        if fix_bgr == "true":
            gen_np = gen_np[:, :, ::-1]

        if ref_np.shape != gen_np.shape:
            ref_np = cv2.resize(ref_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_AREA)

        valid_mask = None
        if ignore_mask is not None:
            mask_np = ignore_mask.cpu().numpy()
            if mask_np.ndim == 3: mask_np = mask_np[0] 
            mask_np = cv2.resize(mask_np, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid_pixels_bool = mask_np < 0.5
        else:
            valid_pixels_bool = np.ones((gen_np.shape[0], gen_np.shape[1]), dtype=bool)

        ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        ref_valid = ref_lab[valid_pixels_bool]
        gen_valid = gen_lab[valid_pixels_bool]

        if len(ref_valid) == 0 or len(gen_valid) == 0:
            print("Warning: Mask covers entire image, using global stats.")
            ref_valid = ref_lab
            gen_valid = gen_lab

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

        img_out = torch.from_numpy(final_rgb).float() / 255.0
        img_out = img_out.unsqueeze(0) 

        return (img_out,)

NODE_CLASS_MAPPINGS = {
    "SmartColorMatch": SmartColorMatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorMatch": "Smart Color Match (Masked)"
}