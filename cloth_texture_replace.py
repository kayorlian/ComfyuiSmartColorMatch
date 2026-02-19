import torch
import numpy as np
import cv2
import mediapipe as mp

class ClothTextureReplace:
    def __init__(self):
        # 初始化人脸检测模型
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_image": ("IMAGE",),      # 模特原图 [B, H, W, C]
                "texture_image": ("IMAGE",),    # 面料素材 [B, H, W, C]
                "mask": ("MASK",),              # 衣服遮罩 [B, H, W]
                "texture_width_cm": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 0.5}),
                "strength": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_cloth"
    CATEGORY = "Image/Texture"

    def _get_pixel_per_cm(self, image):
        """计算每厘米代表多少像素 (PPC)"""
        h, w = image.shape[:2]
        # MediaPipe 需要 RGB
        results = self.face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            # 兜底：如果没有检测到人脸，假设图片宽度对应约 50cm (半身照) 或其他比例
            # 这里简单处理：假设 100px = 1cm 的默认比例，或者基于图片宽度估算
            return w / 50.0 
            
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        face_width_px = bbox.width * w
        
        AVG_FACE_WIDTH_CM = 15.0
        return face_width_px / AVG_FACE_WIDTH_CM

    def _flatten_texture(self, img, kernel_size=None):
        if img is None: return None
        h, w = img.shape[:2]
        img_float = img.astype(np.float32)
        
        if kernel_size is None:
            k = int(min(h, w) / 5)
            k = k | 1 
            if k < 31: k = 31
        else:
            k = kernel_size | 1

        blur = cv2.GaussianBlur(img_float, (k, k), 0)
        blur = np.maximum(blur, 1.0)
        mean_val = np.mean(blur, axis=(0, 1))
        result = (img_float / blur) * mean_val
        return np.clip(result, 0, 255).astype(np.uint8)

    def _make_seamless(self, img, overlap_percent=0.15):
        if overlap_percent <= 0: return img
        h, w = img.shape[:2]
        img_f = img.astype(np.float32)
        
        # X轴
        cut_w = int(w * overlap_percent)
        if cut_w > 0:
            left_part = img_f[:, :cut_w]
            right_part = img_f[:, -cut_w:]
            mask = np.linspace(0, 1, cut_w).reshape(1, cut_w, 1)
            mask = np.tile(mask, (h, 1, 1))
            blended_strip = left_part * mask + right_part * (1.0 - mask)
            img_f[:, :cut_w] = blended_strip
            img_f = img_f[:, :-cut_w] 
            
        # Y轴
        h, w = img_f.shape[:2]
        cut_h = int(h * overlap_percent)
        if cut_h > 0:
            top_part = img_f[:cut_h, :]
            bottom_part = img_f[-cut_h:, :]
            mask = np.linspace(0, 1, cut_h).reshape(cut_h, 1, 1)
            mask = np.tile(mask, (1, w, 1))
            blended_strip = top_part * mask + bottom_part * (1.0 - mask)
            img_f[:cut_h, :] = blended_strip
            img_f = img_f[:-cut_h, :] 

        return img_f.astype(np.uint8)

    def _seamless_tile(self, texture, target_shape):
        th, tw = texture.shape[:2]
        h, w = target_shape[:2]
        repeat_y = int(np.ceil(h / th))
        repeat_x = int(np.ceil(w / tw))
        tiled = np.tile(texture, (repeat_y, repeat_x, 1))
        return tiled[:h, :w]

    def _generate_displacement(self, img_gray, strength=30):
        blurred = cv2.GaussianBlur(img_gray, (21, 21), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        flow_x = grad_x / 255.0 * strength
        flow_y = grad_y / 255.0 * strength
        return flow_x, flow_y

    def _apply_displacement(self, img, flow_x, flow_y):
        h, w = img.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x - flow_x).astype(np.float32)
        map_y = (grid_y - flow_y).astype(np.float32)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def process_one_image(self, model_img_bgr, mask_img_gray, texture_img_bgr, texture_real_width_cm, strength):
        """核心单张图片处理逻辑"""
        h, w = model_img_bgr.shape[:2]
        
        # 0. 预处理纹理
        texture_img = self._flatten_texture(texture_img_bgr)
        
        # 1. 无缝纹理
        texture_seamless = self._make_seamless(texture_img, overlap_percent=0.15)
        
        # 2. 计算缩放并平铺
        ppc = self._get_pixel_per_cm(model_img_bgr)
        target_tex_width_px = int(texture_real_width_cm * ppc)
        
        tex_h, tex_w = texture_seamless.shape[:2]
        scale = target_tex_width_px / tex_w if tex_w > 0 else 1.0
        scale = np.clip(scale, 0.1, 5.0)
        
        new_tex_w, new_tex_h = int(tex_w * scale), int(tex_h * scale)
        # 防止尺寸过小
        if new_tex_w < 1: new_tex_w = 1
        if new_tex_h < 1: new_tex_h = 1
        
        texture_resized = cv2.resize(texture_seamless, (new_tex_w, new_tex_h), interpolation=cv2.INTER_AREA)

        # 3. 分离原图亮度
        img_hsv = cv2.cvtColor(model_img_bgr, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(img_hsv)
        lighting = cv2.GaussianBlur(v_channel, (5, 5), 0)

        # 4. 纹理扭曲
        tiled_texture = self._seamless_tile(texture_resized, (h, w))
        flow_x, flow_y = self._generate_displacement(lighting, strength=strength)
        warped_texture = self._apply_displacement(tiled_texture, flow_x, flow_y)

        # 5. 纹理混合
        warped_texture_gray = cv2.cvtColor(warped_texture, cv2.COLOR_BGR2GRAY)
        tex_float = warped_texture_gray.astype(np.float32) / 255.0
        tex_mean = np.mean(tex_float)
        if tex_mean < 0.01: tex_mean = 0.01
        
        tex_modifier = tex_float / tex_mean
        v_float = v_channel.astype(np.float32)
        v_new = v_float * tex_modifier
        v_new = np.clip(v_new, 0, 255).astype(np.uint8)
        
        result_hsv = cv2.merge([h_channel, s_channel, v_new])
        result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)

        # 6. Mask 融合
        # 确保 Mask 是单通道
        if mask_img_gray.ndim == 3:
            mask_img_gray = cv2.cvtColor(mask_img_gray, cv2.COLOR_BGR2GRAY)

        # 简单的二值化和羽化
        _, mask_bin = cv2.threshold(mask_img_gray, 127, 255, cv2.THRESH_BINARY)
        mask_blur = cv2.GaussianBlur(mask_bin, (7, 7), 0)
        
        mask_float = mask_blur.astype(np.float32) / 255.0
        mask_3c = cv2.merge([mask_float, mask_float, mask_float])

        final_output = (model_img_bgr.astype(np.float32) * (1.0 - mask_3c) + 
                        result_bgr.astype(np.float32) * mask_3c)
        
        return np.clip(final_output, 0, 255).astype(np.uint8)

    def process_cloth(self, model_image, texture_image, mask, texture_width_cm, strength):
        # 准备输出列表
        output_tensors = []
        
        batch_size = model_image.shape[0]
        tex_batch_size = texture_image.shape[0]
        mask_batch_size = mask.shape[0]

        # ComfyUI Mask 是 [B, H, W]，需要检查
        mask_np_all = mask.cpu().numpy()

        for i in range(batch_size):
            # 1. 转换数据类型 ComfyUI (RGB Float 0-1) -> OpenCV (BGR Uint8 0-255)
            
            # Model Image
            curr_model = model_image[i].cpu().numpy()
            curr_model = (curr_model * 255.0).clip(0, 255).astype(np.uint8)
            curr_model_bgr = cv2.cvtColor(curr_model, cv2.COLOR_RGB2BGR)
            
            # Texture Image (循环使用)
            curr_tex = texture_image[i % tex_batch_size].cpu().numpy()
            curr_tex = (curr_tex * 255.0).clip(0, 255).astype(np.uint8)
            curr_tex_bgr = cv2.cvtColor(curr_tex, cv2.COLOR_RGB2BGR)

            # Mask (循环使用)
            # Mask 在 ComfyUI 中通常是 0-1 float
            curr_mask = mask_np_all[i % mask_batch_size]
            # 调整 mask 大小以匹配原图
            if curr_mask.shape != curr_model_bgr.shape[:2]:
                curr_mask = cv2.resize(curr_mask, (curr_model_bgr.shape[1], curr_model_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            curr_mask_uint8 = (curr_mask * 255.0).clip(0, 255).astype(np.uint8)

            # 2. 执行处理
            result_bgr = self.process_one_image(
                curr_model_bgr, 
                curr_mask_uint8, 
                curr_tex_bgr, 
                texture_width_cm, 
                strength
            )

            # 3. 转回 ComfyUI 格式 (BGR -> RGB -> Float)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0)
            output_tensors.append(result_tensor)

        final_tensor = torch.stack(output_tensors, dim=0)
        return (final_tensor,)

NODE_CLASS_MAPPINGS = {
    "ClothTextureReplace": ClothTextureReplace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClothTextureReplace": "Cloth Texture Replace (Masked)"
}