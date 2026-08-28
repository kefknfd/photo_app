import os
import sys
import io
import time
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["U2NET_HOME"] = str(BASE_DIR)

import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageEnhance

SESSION = None

# --- 智慧去背處理模組 ---

def find_u2net_model():
    search_dirs = [
        BASE_DIR,
        Path.cwd(),
        Path.home() / ".u2net"
    ]
    for d in search_dirs:
        target = d / "u2net.onnx"
        if target.is_file():
            return str(target)
        target_sub = d / ".u2net" / "u2net.onnx"
        if target_sub.is_file():
            return str(target_sub)
    return None

def get_rembg_session():
    global SESSION
    if SESSION is not None:
        return SESSION

    model_path = find_u2net_model()
    if not model_path:
        SESSION = False
        return False

    try:
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        SESSION = ort.InferenceSession(model_path, sess_opts, providers=['CPUExecutionProvider'])
        return SESSION
    except Exception:
        SESSION = False
        return False

def apply_background_matting(processed_pil, orig_pil, bg_mode, feather_val):
    if bg_mode == "保留原圖背景":
        return processed_pil

    session = get_rembg_session()
    if not session:
        return processed_pil

    try:
        img_rgb = np.array(orig_pil.convert("RGB"))
        h_orig, w_orig = img_rgb.shape[:2]
        
        resized = cv2.resize(img_rgb, (320, 320)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (resized - mean) / std
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        pred = session.run([output_name], {input_name: input_tensor})[0]

        raw_mask = pred[0, 0, :, :]
        raw_mask = (raw_mask - raw_mask.min()) / (raw_mask.max() - raw_mask.min() + 1e-8)

        mask_full = cv2.resize(raw_mask, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        thresh_val = np.clip(0.3 + (1.0 - feather_val) * 0.4, 0.1, 0.9)
        
        alpha = np.zeros_like(mask_full, dtype=np.float32)
        alpha[mask_full >= thresh_val] = 1.0
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        alpha_3d = alpha[:, :, np.newaxis]
    except Exception:
        return processed_pil

    proc_rgb = np.array(processed_pil.convert("RGB"))

    if bg_mode == "✂️ 透明背景 (PNG)":
        proc_rgba = np.dstack([proc_rgb, (alpha * 255).astype(np.uint8)])
        return Image.fromarray(proc_rgba, "RGBA")

    bg_colors = {
        "⚪ 純白證件照背景": np.array([255, 255, 255], dtype=np.uint8),
        "🔵 商務證件藍背景": np.array([67, 142, 219], dtype=np.uint8),
        "🔴 喜慶證件紅背景": np.array([218, 41, 28], dtype=np.uint8),
        "🔘 質感冷灰背景": np.array([220, 222, 225], dtype=np.uint8),
        "🍵 莫蘭迪綠背景": np.array([178, 190, 181], dtype=np.uint8)
    }

    if bg_mode in bg_colors:
        bg_rgb = np.full_like(proc_rgb, bg_colors[bg_mode])
        composed = (proc_rgb * alpha_3d + bg_rgb * (1.0 - alpha_3d)).astype(np.uint8)
        return Image.fromarray(composed, "RGB")
    elif bg_mode == "🖤 背景黑白 (人物全彩聚焦)":
        gray_bg = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray_bg_rgb = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB)
        composed = (proc_rgb * alpha_3d + gray_bg_rgb * (1.0 - alpha_3d)).astype(np.uint8)
        return Image.fromarray(composed, "RGB")
    elif bg_mode == "📸 背景大光圈深層虛化 (Bokeh)":
        blurred_bg = cv2.GaussianBlur(img_rgb, (51, 51), 0)
        composed = (proc_rgb * alpha_3d + blurred_bg * (1.0 - alpha_3d)).astype(np.uint8)
        return Image.fromarray(composed, "RGB")

    return processed_pil

ART_STYLE_CHOICES = [
    "原圖無藝術濾鏡",
    "🎨 經典卡通風 (Cartoon)",
    "🌸 日系動漫風 (Anime)",
    "🖌️ 印象派油畫 (Oil Painting)",
    "🎨 夢幻水彩畫 (Watercolor)",
    "🏮 單色墨水渲染 (Ink Wash Sketch)",
    "🏮 東方水墨風 (Ink Wash)",
    "✏️ 復古素描風 (Pencil Sketch)",
    "🌆 賽博龐克霓虹 (Cyberpunk)",
    "👾 復古像素藝術 (Pixel Art)",
    "🎨 普普藝術風 (Pop Art)"
]

DEFAULTS = {
    "bg_mode": "保留原圖背景",
    "bg_feather": 0.5,
    "art_style": "原圖無藝術濾鏡",
    "art_blend": 1.0,
    "line_strength": 0.5,
    "cel_shading": 0.5,
    "smooth": 0.0,
    "matte": 0.0,
    "detail": 0.0,
    "eye_clarity": 0.0,
    "teeth_white": 0.0,
    "blush": 0.0,
    "glow": 0.0,
    "bokeh": 0.0,
    "shadow": 0.0,
    "highlight": 0.0,
    "brightness": 1.00,
    "contrast": 1.00,
    "saturation": 1.00,
    "sharpness": 1.00,
    "temp": 0.0,
    "grain": 0.0,
    "vignette": 0.0,
    "filter": "經典無濾鏡",
    "rotation": "0°",
    "flip_h": False
}

PRESETS = {
    "🌸 膠原蛋白裸妝": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.15, "matte": 0.0, "detail": 0.05, "eye_clarity": 0.10, "teeth_white": 0.0,
        "blush": 0.08, "glow": 0.10, "bokeh": 0.0, "shadow": 0.05, "highlight": 0.05,
        "brightness": 1.03, "contrast": 1.01, "saturation": 1.02, "sharpness": 1.02,
        "temp": 0.02, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"
    },
    "☀️ 自然暖色調": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.15, "matte": 0.0, "detail": 0.05, "eye_clarity": 0.10, "teeth_white": 0.05,
        "blush": 0.10, "glow": 0.08, "bokeh": 0.0, "shadow": 0.05, "highlight": 0.05,
        "brightness": 1.03, "contrast": 1.02, "saturation": 1.05, "sharpness": 1.02,
        "temp": 0.18, "grain": 0.0, "vignette": 0.05, "filter": "經典無濾鏡"
    },
    "☁️ 柔和對比": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.20, "matte": 0.0, "detail": 0.0, "eye_clarity": 0.05, "teeth_white": 0.05,
        "blush": 0.05, "glow": 0.20, "bokeh": 0.05, "shadow": 0.15, "highlight": 0.12,
        "brightness": 1.04, "contrast": 0.90, "saturation": 0.96, "sharpness": 0.95,
        "temp": 0.0, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"
    },
    "🌟 雜誌高訂啞光": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.20, "matte": 0.30, "detail": 0.10, "eye_clarity": 0.20, "teeth_white": 0.20,
        "blush": 0.05, "glow": 0.0, "bokeh": 0.10, "shadow": 0.05, "highlight": 0.15,
        "brightness": 1.01, "contrast": 1.05, "saturation": 0.98, "sharpness": 1.10,
        "temp": 0.0, "grain": 0.05, "vignette": 0.10, "filter": "經典無濾鏡"
    },
    "✨ 韓系清透冷白": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.20, "matte": 0.10, "detail": 0.05, "eye_clarity": 0.15, "teeth_white": 0.20,
        "blush": 0.05, "glow": 0.15, "bokeh": 0.05, "shadow": 0.10, "highlight": 0.05,
        "brightness": 1.05, "contrast": 0.99, "saturation": 0.98, "sharpness": 1.02,
        "temp": -0.10, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"
    },
    "🌿 日系清新空氣感": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.15, "matte": 0.0, "detail": 0.05, "eye_clarity": 0.10, "teeth_white": 0.10,
        "blush": 0.05, "glow": 0.10, "bokeh": 0.05, "shadow": 0.10, "highlight": 0.05,
        "brightness": 1.04, "contrast": 0.98, "saturation": 0.95, "sharpness": 1.02,
        "temp": -0.08, "grain": 0.0, "vignette": 0.0, "filter": "🎞️ 富士正片膠卷 (Fuji Film)"
    },
    "☕ 復古法式暖咖": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.15, "matte": 0.05, "detail": 0.05, "eye_clarity": 0.10, "teeth_white": 0.10,
        "blush": 0.12, "glow": 0.10, "bokeh": 0.05, "shadow": 0.05, "highlight": 0.10,
        "brightness": 0.99, "contrast": 1.04, "saturation": 0.98, "sharpness": 1.05,
        "temp": 0.20, "grain": 0.15, "vignette": 0.25, "filter": "🌅 柯達金感暖調 (Kodak Gold)"
    },
    "🧸 復古微醺暖調": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.15, "matte": 0.0, "detail": 0.05, "eye_clarity": 0.10, "teeth_white": 0.10,
        "blush": 0.15, "glow": 0.10, "bokeh": 0.05, "shadow": 0.05, "highlight": 0.10,
        "brightness": 0.99, "contrast": 1.02, "saturation": 1.02, "sharpness": 1.05,
        "temp": 0.15, "grain": 0.10, "vignette": 0.15, "filter": "🌅 柯達金感暖調 (Kodak Gold)"
    },
    "🍷 復古法式紅酒": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.15, "matte": 0.10, "detail": 0.10, "eye_clarity": 0.15, "teeth_white": 0.15,
        "blush": 0.20, "glow": 0.10, "bokeh": 0.10, "shadow": 0.05, "highlight": 0.10,
        "brightness": 0.98, "contrast": 1.08, "saturation": 1.05, "sharpness": 1.08,
        "temp": 0.12, "grain": 0.10, "vignette": 0.25, "filter": "🍷 法式紅酒 (Bordeaux Mood)"
    },
    "🏙️ 俐落冷萃都會": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.10, "matte": 0.15, "detail": 0.15, "eye_clarity": 0.20, "teeth_white": 0.20,
        "blush": 0.0, "glow": 0.0, "bokeh": 0.10, "shadow": 0.0, "highlight": 0.10,
        "brightness": 1.00, "contrast": 1.10, "saturation": 0.85, "sharpness": 1.15,
        "temp": -0.15, "grain": 0.05, "vignette": 0.10, "filter": "經典無濾鏡"
    },
    "🌊 冰島冷萃清冽": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.15, "matte": 0.10, "detail": 0.15, "eye_clarity": 0.25, "teeth_white": 0.25,
        "blush": 0.05, "glow": 0.05, "bokeh": 0.10, "shadow": 0.10, "highlight": 0.10,
        "brightness": 1.03, "contrast": 1.05, "saturation": 0.88, "sharpness": 1.15,
        "temp": -0.25, "grain": 0.05, "vignette": 0.10, "filter": "🌊 冷萃青藍 (Nordic Cool)"
    },
    "🎬 電影青橙色調": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.10, "matte": 0.0, "detail": 0.10, "eye_clarity": 0.15, "teeth_white": 0.15,
        "blush": 0.10, "glow": 0.0, "bokeh": 0.10, "shadow": 0.05, "highlight": 0.10,
        "brightness": 1.00, "contrast": 1.08, "saturation": 1.05, "sharpness": 1.10,
        "temp": 0.05, "grain": 0.10, "vignette": 0.15, "filter": "🎬 電影青橙 (Teal & Orange)"
    },
    "🖤 黑白經典紀實": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.10, "matte": 0.0, "detail": 0.15, "eye_clarity": 0.20, "teeth_white": 0.10,
        "blush": 0.0, "glow": 0.0, "bokeh": 0.10, "shadow": 0.05, "highlight": 0.05,
        "brightness": 1.01, "contrast": 1.10, "saturation": 1.00, "sharpness": 1.15,
        "temp": 0.0, "grain": 0.15, "vignette": 0.10, "filter": "🖤 高對比黑白藝術 (Monochrome)"
    },
    "💡 逆光人像救援": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5,
        "smooth": 0.10, "matte": 0.0, "detail": 0.05, "eye_clarity": 0.10, "teeth_white": 0.10,
        "blush": 0.05, "glow": 0.0, "bokeh": 0.0, "shadow": 0.25, "highlight": 0.20,
        "brightness": 1.02, "contrast": 1.01, "saturation": 1.02, "sharpness": 1.05,
        "temp": 0.02, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"
    },
    "🎨 經典卡通風 (Cartoon)": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "🎨 經典卡通風 (Cartoon)", "art_blend": 0.85, "line_strength": 0.60, "cel_shading": 0.60,
        "smooth": 0.10, "matte": 0.0, "detail": 0.10, "eye_clarity": 0.10, "teeth_white": 0.0,
        "blush": 0.0, "glow": 0.0, "bokeh": 0.0, "shadow": 0.0, "highlight": 0.0,
        "brightness": 1.05, "contrast": 1.10, "saturation": 1.15, "sharpness": 1.10,
        "temp": 0.0, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"
    },
    "🌸 日系動漫風 (Anime)": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "🌸 日系動漫風 (Anime)", "art_blend": 0.90, "line_strength": 0.50, "cel_shading": 0.50,
        "smooth": 0.20, "matte": 0.0, "detail": 0.05, "eye_clarity": 0.20, "teeth_white": 0.10,
        "blush": 0.15, "glow": 0.15, "bokeh": 0.0, "shadow": 0.10, "highlight": 0.05,
        "brightness": 1.08, "contrast": 1.02, "saturation": 1.15, "sharpness": 1.05,
        "temp": -0.05, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"
    },
    "🖌️ 印象派油畫 (Oil Painting)": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "🖌️ 印象派油畫 (Oil Painting)", "art_blend": 0.85, "line_strength": 0.30, "cel_shading": 0.50,
        "smooth": 0.05, "matte": 0.0, "detail": 0.10, "eye_clarity": 0.10, "teeth_white": 0.0,
        "blush": 0.10, "glow": 0.05, "bokeh": 0.05, "shadow": 0.05, "highlight": 0.10,
        "brightness": 1.00, "contrast": 1.08, "saturation": 1.20, "sharpness": 1.10,
        "temp": 0.10, "grain": 0.10, "vignette": 0.15, "filter": "🌅 柯達金感暖調 (Kodak Gold)"
    },
    "🎨 夢幻水彩畫 (Watercolor)": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "🎨 夢幻水彩畫 (Watercolor)", "art_blend": 0.85, "line_strength": 0.35, "cel_shading": 0.50,
        "smooth": 0.15, "matte": 0.0, "detail": 0.05, "eye_clarity": 0.10, "teeth_white": 0.0,
        "blush": 0.10, "glow": 0.10, "bokeh": 0.05, "shadow": 0.05, "highlight": 0.05,
        "brightness": 1.04, "contrast": 1.02, "saturation": 1.10, "sharpness": 1.00,
        "temp": 0.0, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"
    },
    "🏮 單色墨水渲染 (Ink Wash Sketch)": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "🏮 單色墨水渲染 (Ink Wash Sketch)", "art_blend": 0.95, "line_strength": 0.65, "cel_shading": 0.70,
        "smooth": 0.05, "matte": 0.0, "detail": 0.15, "eye_clarity": 0.20, "teeth_white": 0.0,
        "blush": 0.0, "glow": 0.0, "bokeh": 0.0, "shadow": 0.05, "highlight": 0.05,
        "brightness": 1.02, "contrast": 1.12, "saturation": 0.0, "sharpness": 1.15,
        "temp": 0.0, "grain": 0.10, "vignette": 0.15, "filter": "經典無濾鏡"
    },
    "🏮 東方水墨風 (Ink Wash)": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "🏮 東方水墨風 (Ink Wash)", "art_blend": 0.90, "line_strength": 0.40, "cel_shading": 0.70,
        "smooth": 0.0, "matte": 0.0, "detail": 0.05, "eye_clarity": 0.0, "teeth_white": 0.0,
        "blush": 0.0, "glow": 0.0, "bokeh": 0.0, "shadow": 0.0, "highlight": 0.0,
        "brightness": 1.00, "contrast": 1.05, "saturation": 1.00, "sharpness": 1.00,
        "temp": 0.0, "grain": 0.05, "vignette": 0.15, "filter": "經典無濾鏡"
    },
    "✏️ 復古素描風 (Pencil Sketch)": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "✏️ 復古素描風 (Pencil Sketch)", "art_blend": 0.95, "line_strength": 0.60, "cel_shading": 0.50,
        "smooth": 0.0, "matte": 0.0, "detail": 0.15, "eye_clarity": 0.0, "teeth_white": 0.0,
        "blush": 0.0, "glow": 0.0, "bokeh": 0.0, "shadow": 0.0, "highlight": 0.0,
        "brightness": 1.05, "contrast": 1.10, "saturation": 1.00, "sharpness": 1.20,
        "temp": 0.0, "grain": 0.15, "vignette": 0.10, "filter": "經典無濾鏡"
    },
    "🌆 賽博龐克霓虹 (Cyberpunk)": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "🌆 賽博龐克霓虹 (Cyberpunk)", "art_blend": 0.90, "line_strength": 0.50, "cel_shading": 0.60,
        "smooth": 0.10, "matte": 0.0, "detail": 0.25, "eye_clarity": 0.30, "teeth_white": 0.0,
        "blush": 0.0, "glow": 0.20, "bokeh": 0.10, "shadow": 0.15, "highlight": 0.15,
        "brightness": 1.02, "contrast": 1.20, "saturation": 1.30, "sharpness": 1.25,
        "temp": -0.20, "grain": 0.15, "vignette": 0.25, "filter": "🎬 電影青橙 (Teal & Orange)"
    },
    "👾 復古像素藝術 (Pixel Art)": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "👾 復古像素藝術 (Pixel Art)", "art_blend": 1.0, "line_strength": 0.50, "cel_shading": 0.50,
        "smooth": 0.0, "matte": 0.0, "detail": 0.0, "eye_clarity": 0.0, "teeth_white": 0.0,
        "blush": 0.0, "glow": 0.0, "bokeh": 0.0, "shadow": 0.0, "highlight": 0.0,
        "brightness": 1.05, "contrast": 1.15, "saturation": 1.20, "sharpness": 1.00,
        "temp": 0.0, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"
    },
    "🎨 普普藝術風 (Pop Art)": {
        "bg_mode": "保留原圖背景", "bg_feather": 0.5,
        "art_style": "🎨 普普藝術風 (Pop Art)", "art_blend": 0.90, "line_strength": 0.65, "cel_shading": 0.75,
        "smooth": 0.0, "matte": 0.0, "detail": 0.15, "eye_clarity": 0.10, "teeth_white": 0.0,
        "blush": 0.0, "glow": 0.0, "bokeh": 0.0, "shadow": 0.0, "highlight": 0.0,
        "brightness": 1.08, "contrast": 1.25, "saturation": 1.40, "sharpness": 1.15,
        "temp": 0.05, "grain": 0.05, "vignette": 0.0, "filter": "經典無濾鏡"
    }
}

def render_art_style(img_bgr, style_name, blend_ratio, line_strength, cel_shading):
    if style_name == "原圖無藝術濾鏡" or blend_ratio <= 0:
        return img_bgr

    art_bgr = img_bgr.copy()

    if style_name == "🎨 普普藝術風 (Pop Art)":
        smooth = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=100, sigmaSpace=100)
        div = max(int(64 + (1.0 - cel_shading) * 64), 32)
        quantized = (smooth // div) * div + div // 2
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        c_val = 2 + int(line_strength * 4)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, c_val)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        pop_base = cv2.bitwise_and(quantized, edges_bgr)
        hsv = cv2.cvtColor(pop_base, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.45 + 15, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.10, 0, 255)
        art_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif style_name == "🏮 單色墨水渲染 (Ink Wash Sketch)":
        smooth = img_bgr.copy()
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, d=7, sigmaColor=90, sigmaSpace=90)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        div = max(int(48 + (1.0 - cel_shading) * 48), 24)
        quantized = (gray // div) * div + div // 2
        quantized_bgr = cv2.merge([quantized, quantized, quantized])
        blurred = cv2.GaussianBlur(quantized_bgr, (17, 17), 0)
        bled = cv2.addWeighted(quantized_bgr, 0.65, blurred, 0.35, 0)
        gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred_orig = cv2.medianBlur(gray_orig, 3)
        c_val = 2 + int(line_strength * 3)
        line_mask = cv2.adaptiveThreshold(
            blurred_orig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, c_val
        )
        line_mask = cv2.erode(line_mask, np.ones((2, 2), np.uint8), iterations=1)
        line_bgr = cv2.merge([line_mask, line_mask, line_mask])
        art_bgr = cv2.bitwise_and(bled, line_bgr)

    elif style_name == "🌸 日系動漫風 (Anime)":
        smooth = img_bgr.copy()
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, d=7, sigmaColor=75, sigmaSpace=75)
        div = max(int(32 + (1.0 - cel_shading) * 32), 16)
        quantized = (smooth // div) * div + div // 2
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.medianBlur(gray, 3)
        c_val = 2 + int(line_strength * 3)
        line_mask = cv2.adaptiveThreshold(
            blurred_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, c_val
        )
        line_mask = cv2.erode(line_mask, np.ones((2, 2), np.uint8), iterations=1)
        line_bgr = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
        anime_base = cv2.bitwise_and(quantized, line_bgr)
        hsv = cv2.cvtColor(anime_base, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.28 + 5, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.08 + 15, 0, 255)
        anime_colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        highlight_glow = cv2.GaussianBlur(anime_colored, (15, 15), 0)
        art_bgr = cv2.addWeighted(anime_colored, 0.85, highlight_glow, 0.15, 0)

    elif style_name == "🎨 經典卡通風 (Cartoon)":
        smooth = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=90, sigmaSpace=90)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        c_val = 2 + int(line_strength * 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, c_val)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        div = max(int(40 + (1.0 - cel_shading) * 40), 20)
        quantized = (smooth // div) * div + div // 2
        art_bgr = cv2.bitwise_and(quantized, edges_color)

    elif style_name == "🖌️ 印象派油畫 (Oil Painting)":
        try:
            art_bgr = cv2.xphoto.oilPainting(img_bgr, 4, 1)
        except Exception:
            art_bgr = cv2.stylization(img_bgr, sigma_s=60, sigma_r=0.45)

    elif style_name == "🎨 夢幻水彩畫 (Watercolor)":
        smooth = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=200, sigmaSpace=200)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edge = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
        edge_inv = cv2.bitwise_not(edge)
        edge_bgr = cv2.cvtColor(edge_inv, cv2.COLOR_GRAY2BGR)
        art_bgr = cv2.addWeighted(smooth, 0.85, edge_bgr, 0.15, 0)

    elif style_name == "🏮 東方水墨風 (Ink Wash)":
        smooth = img_bgr.copy()
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=100, sigmaSpace=100)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        quantized = np.floor(gray / 64.0) * 85.0
        quantized = cv2.GaussianBlur(quantized.astype(np.uint8), (5, 5), 0)
        edges = cv2.Canny(gray, 30, 90)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        ink = np.clip(quantized.astype(np.float32) - (edges.astype(np.float32) * 0.7), 0, 255).astype(np.uint8)
        paper = np.zeros_like(img_bgr, dtype=np.float32)
        paper[:, :, 0] = ink * 0.90
        paper[:, :, 1] = ink * 0.96
        paper[:, :, 2] = ink * 1.00
        art_bgr = np.clip(paper, 0, 255).astype(np.uint8)

    elif style_name == "✏️ 復古素描風 (Pencil Sketch)":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (31, 31), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=250)
        art_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif style_name == "🌆 賽博龐克霓虹 (Cyberpunk)":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 100)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        neon = np.zeros_like(img_bgr)
        neon[:, :, 0] = edges * 1.0
        neon[:, :, 2] = edges * 0.9
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 80) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        color_shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        art_bgr = cv2.addWeighted(color_shifted, 0.75, neon, 0.4, 0)

    elif style_name == "👾 復古像素藝術 (Pixel Art)":
        h, w = img_bgr.shape[:2]
        pixel_size = max(int(min(h, w) / 90), 4)
        temp = cv2.resize(img_bgr, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        art_bgr = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    return cv2.addWeighted(art_bgr, blend_ratio, img_bgr, 1.0 - blend_ratio, 0)

# --- 基礎人像與光影調整演算法 ---

def apply_skin_matte(img_bgr, strength):
    if strength <= 0:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    shine_mask = cv2.threshold(v_channel, 215, 255, cv2.THRESH_BINARY)[1]
    shine_mask = cv2.GaussianBlur(shine_mask, (31, 31), 0) / 255.0
    img_float = img_bgr.astype(np.float32)
    img_float -= (img_float * 0.12) * shine_mask[:, :, np.newaxis] * strength
    return np.clip(img_float, 0, 255).astype(np.uint8)

def apply_teeth_whitening(img_bgr, strength):
    if strength <= 0:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    yellow_mask = cv2.inRange(hsv, np.array([18, 30, 140]), np.array([35, 180, 255]))
    yellow_mask = cv2.GaussianBlur(yellow_mask, (11, 11), 0) / 255.0
    s_new = np.clip(s.astype(np.float32) - (s * strength * 0.5) * yellow_mask, 0, 255).astype(np.uint8)
    v_new = np.clip(v.astype(np.float32) + (20 * strength) * yellow_mask, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s_new, v_new]), cv2.COLOR_HSV2BGR)

def apply_eye_clarity(img_bgr, strength):
    if strength <= 0:
        return img_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    high_freq = cv2.Laplacian(gray, cv2.CV_32F)
    high_freq = np.clip(high_freq, -15, 15)[:, :, np.newaxis]
    img_float = img_bgr.astype(np.float32) + high_freq * strength * 0.4
    return np.clip(img_float, 0, 255).astype(np.uint8)

def apply_bokeh_blur(img_bgr, strength):
    if strength <= 0:
        return img_bgr
    rows, cols = img_bgr.shape[:2]
    blurred = cv2.GaussianBlur(img_bgr, (25, 25), 0)
    kernel_x = cv2.getGaussianKernel(cols, cols / 2.0)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2.0)
    mask = (kernel_y * kernel_x.T)
    mask = mask / mask.max()
    mask = np.clip(mask * (1.0 + (1.0 - strength)), 0, 1)[:, :, np.newaxis]
    bokeh = img_bgr.astype(np.float32) * mask + blurred.astype(np.float32) * (1.0 - mask)
    return np.clip(bokeh, 0, 255).astype(np.uint8)

def apply_skin_blush(img_bgr, blush_strength):
    if blush_strength <= 0:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, np.array([0, 25, 100]), np.array([20, 180, 255]))
    skin_mask = cv2.GaussianBlur(skin_mask, (25, 25), 0) / 255.0
    img_float = img_bgr.astype(np.float32)
    img_float[:, :, 2] += skin_mask * blush_strength * 10.0
    return np.clip(img_float, 0, 255).astype(np.uint8)

def apply_orton_glow(img_bgr, glow_strength):
    if glow_strength <= 0:
        return img_bgr
    blurred = cv2.GaussianBlur(img_bgr, (21, 21), 0)
    base = img_bgr.astype(np.float32) / 255.0
    blur = blurred.astype(np.float32) / 255.0
    screen = 1.0 - (1.0 - base) * (1.0 - blur)
    result = base * (1.0 - glow_strength) + screen * glow_strength
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)

def apply_film_grain(img_bgr, grain_strength):
    if grain_strength <= 0:
        return img_bgr
    noise = np.random.normal(0, grain_strength * 15, img_bgr.shape).astype(np.float32)
    return np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def apply_vignette(img_bgr, strength):
    if strength <= 0:
        return img_bgr
    rows, cols = img_bgr.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2)
    mask = (kernel_y * kernel_x.T)
    mask = 1 - strength * (1 - (mask / mask.max()))
    vignette = np.empty_like(img_bgr)
    for i in range(3):
        vignette[:, :, i] = np.clip(img_bgr[:, :, i] * mask, 0, 255)
    return vignette.astype(np.uint8)

def adjust_shadows_highlights(img_bgr, shadow_lift, highlight_suppress):
    if shadow_lift == 0 and highlight_suppress == 0:
        return img_bgr
    img_float = img_bgr.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow_mask = np.clip(1.0 - (gray / 0.4), 0, 1)[:, :, np.newaxis]
    highlight_mask = np.clip((gray - 0.6) / 0.4, 0, 1)[:, :, np.newaxis]
    if shadow_lift > 0:
        img_float += shadow_mask * shadow_lift * 0.15
    if highlight_suppress > 0:
        img_float -= highlight_mask * highlight_suppress * 0.15
    return np.clip(img_float * 255.0, 0, 255).astype(np.uint8)

def apply_film_filter(img_bgr, filter_type):
    if filter_type == "經典無濾鏡":
        return img_bgr
    elif filter_type == "🎞️ 富士正片膠卷 (Fuji Film)":
        b, g, r = cv2.split(img_bgr.astype(np.float32))
        b = np.clip(b * 1.03 + 2, 0, 255)
        g = np.clip(g * 1.02 + 1, 0, 255)
        r = np.clip(r * 0.97, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)
    elif filter_type == "🌅 柯達金感暖調 (Kodak Gold)":
        b, g, r = cv2.split(img_bgr.astype(np.float32))
        b = np.clip(b * 0.94, 0, 255)
        g = np.clip(g * 1.01 + 2, 0, 255)
        r = np.clip(r * 1.04 + 4, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)
    elif filter_type == "🎬 電影青橙 (Teal & Orange)":
        b, g, r = cv2.split(img_bgr.astype(np.float32))
        b = np.clip(b * 1.08, 0, 255)
        r = np.clip(r * 1.06 + 2, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)
    elif filter_type == "🍷 法式紅酒 (Bordeaux Mood)":
        b, g, r = cv2.split(img_bgr.astype(np.float32))
        b = np.clip(b * 0.95 + 4, 0, 255)
        r = np.clip(r * 1.15 + 6, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)
    elif filter_type == "🌊 冷萃青藍 (Nordic Cool)":
        b, g, r = cv2.split(img_bgr.astype(np.float32))
        b = np.clip(b * 1.18 + 8, 0, 255)
        g = np.clip(g * 1.05, 0, 255)
        r = np.clip(r * 0.88, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)
    elif filter_type == "🖤 高對比黑白藝術 (Monochrome)":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lut = np.array([np.clip(128 + 1.25 * (i - 128), 0, 255) for i in range(256)]).astype(np.uint8)
        gray_contrast = cv2.LUT(gray, lut)
        return cv2.merge([gray_contrast, gray_contrast, gray_contrast])
    return img_bgr

def process_full_studio(
    input_img,
    bg_mode_choice, bg_feather_val,
    art_style_choice, art_blend_val, line_strength_val, cel_shading_val,
    smooth_val, matte_val, detail_val, eye_val, teeth_val, blush_val, glow_val, bokeh_val,
    shadow_val, highlight_val, brightness_val, contrast_val,
    saturation_val, sharpness_val, temp_val,
    grain_val, vignette_val, filter_choice,
    rotation_choice, flip_choice
):
    if input_img is None:
        return None
    # 🌟 雲端極速優化：將超大手機相片等比例縮放至最高 1600px，大幅提升運算速度
    max_dimension = 1600
    w_orig, h_orig = input_img.size
    if max(w_orig, h_orig) > max_dimension:
        scale = max_dimension / max(w_orig, h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        input_img = input_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img_bgr = cv2.cvtColor(np.array(input_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    if rotation_choice == "90° 順時針":
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_choice == "180° 翻轉":
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)
    elif rotation_choice == "270° 逆時針":
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip_choice:
        img_bgr = cv2.flip(img_bgr, 1)

    img_bgr = render_art_style(img_bgr, art_style_choice, art_blend_val, line_strength_val, cel_shading_val)

    if smooth_val > 0:
        d = int(smooth_val * 10) + 1
        smooth_bgr = cv2.bilateralFilter(img_bgr, d=d, sigmaColor=smooth_val * 40, sigmaSpace=smooth_val * 40)
        img_bgr = cv2.addWeighted(smooth_bgr, smooth_val * 0.5, img_bgr, 1 - (smooth_val * 0.5), 0)

    if matte_val > 0:
        img_bgr = apply_skin_matte(img_bgr, matte_val)
    if teeth_val > 0:
        img_bgr = apply_teeth_whitening(img_bgr, teeth_val)
    if eye_val > 0:
        img_bgr = apply_eye_clarity(img_bgr, eye_val)

    if detail_val > 0:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0 + detail_val * 1.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if blush_val > 0:
        img_bgr = apply_skin_blush(img_bgr, blush_val)
    if glow_val > 0:
        img_bgr = apply_orton_glow(img_bgr, glow_val)
    if bokeh_val > 0:
        img_bgr = apply_bokeh_blur(img_bgr, bokeh_val)
    if shadow_val > 0 or highlight_val > 0:
        img_bgr = adjust_shadows_highlights(img_bgr, shadow_val, highlight_val)

    if temp_val != 0:
        img_float = img_bgr.astype(np.float32)
        if temp_val > 0:
            img_float[:, :, 2] += temp_val * 10
            img_float[:, :, 0] -= temp_val * 5
        else:
            img_float[:, :, 0] += abs(temp_val) * 10
            img_float[:, :, 2] -= abs(temp_val) * 5
        img_bgr = np.clip(img_float, 0, 255).astype(np.uint8)

    if filter_choice != "經典無濾鏡":
        img_bgr = apply_film_filter(img_bgr, filter_choice)
    if grain_val > 0:
        img_bgr = apply_film_grain(img_bgr, grain_val)
    if vignette_val > 0:
        img_bgr = apply_vignette(img_bgr, vignette_val)

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if brightness_val != 1.0:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness_val)
    if contrast_val != 1.0:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast_val)
    if saturation_val != 1.0:
        img_pil = ImageEnhance.Color(img_pil).enhance(saturation_val)
    if sharpness_val != 1.0:
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(sharpness_val)

    final_output = apply_background_matting(img_pil, input_img, bg_mode_choice, bg_feather_val)
    return final_output

def export_file_on_demand(current_img, file_format):
    if current_img is None:
        gr.Warning("⚠️ 尚未有任何精修成果可供下載！")
        return None
        
    timestamp = int(time.time())
    if getattr(current_img, "mode", "RGB") == "RGBA" or file_format.lower() == "png":
        save_path = str(OUTPUT_DIR / f"enhanced_{timestamp}.png")
        current_img.save(save_path, format="PNG")
    else:
        save_path = str(OUTPUT_DIR / f"enhanced_{timestamp}.jpg")
        current_img.convert("RGB").save(save_path, format="JPEG", quality=95)
    
    return gr.update(value=save_path, visible=True)

def on_batch_files_uploaded(file_list):
    if not file_list or len(file_list) == 0:
        return None, None, [], "請先上傳圖片！", gr.update(visible=False)
    
    gallery_items = []
    for idx, f in enumerate(file_list):
        f_path = f.name if hasattr(f, "name") else str(f)
        gallery_items.append((f_path, f"第 {idx+1} 張: {Path(f_path).name}"))
        
    first_path = gallery_items[0][0]
    first_pil = Image.open(first_path).convert("RGB")
    
    status_msg = f"📂 已成功載入 {len(file_list)} 張照片。已載入第 1 張為預覽範本，可點擊下方縮圖切換！"
    return first_pil, first_pil, gallery_items, status_msg, gr.update(visible=False)

def on_gallery_select(evt: gr.SelectData, file_list, *current_params):
    if not file_list or len(file_list) == 0:
        return gr.update(), gr.update(), "無可選取的圖片"
    
    idx = evt.index
    selected_file = file_list[idx]
    selected_path = selected_file.name if hasattr(selected_file, "name") else str(selected_file)
    selected_pil = Image.open(selected_path).convert("RGB")
    
    preview_res = process_full_studio(selected_pil, *current_params)
    status_msg = f"🎯 已切換範本為第 {idx+1} 張 ({Path(selected_path).name})！"
    return selected_pil, preview_res, status_msg

def run_batch_export(file_list, *current_params):
    if not file_list or len(file_list) == 0:
        gr.Warning("⚠️ 尚未上傳任何圖片！")
        return gr.update(visible=False), "請先上傳圖片！"
    
    timestamp = int(time.time())
    zip_filename = f"batch_enhanced_{timestamp}.zip"
    zip_path = str(OUTPUT_DIR / zip_filename)
    
    total = len(file_list)
    processed_count = 0
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, file_obj in enumerate(file_list):
            try:
                src_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
                orig_pil = Image.open(src_path).convert("RGB")
                
                res_pil = process_full_studio(orig_pil, *current_params)
                
                stem = Path(src_path).stem
                if getattr(res_pil, "mode", "RGB") == "RGBA":
                    out_ext = ".png"
                    img_bytes = io.BytesIO()
                    res_pil.save(img_bytes, format="PNG")
                else:
                    out_ext = ".jpg"
                    img_bytes = io.BytesIO()
                    res_pil.convert("RGB").save(img_bytes, format="JPEG", quality=95)
                
                zipf.writestr(f"{stem}_enhanced{out_ext}", img_bytes.getvalue())
                processed_count += 1
            except Exception as e:
                print(f"處理第 {idx+1} 張圖片失敗: {e}")
                continue

    status_msg = f"🎉 全部處理完成！成功打包 {processed_count} / {total} 張照片，請點擊下方按鈕下載 ZIP 檔案。"
    return gr.update(value=zip_path, visible=True), status_msg

def load_preset_universal(preset_name, single_img, batch_sample_img, current_rot, current_flip):
    cfg = PRESETS[preset_name]
    
    single_res = None
    if single_img is not None:
        single_res = process_full_studio(
            single_img,
            cfg["bg_mode"], cfg["bg_feather"],
            cfg["art_style"], cfg["art_blend"], cfg["line_strength"], cfg["cel_shading"],
            cfg["smooth"], cfg["matte"], cfg["detail"], cfg["eye_clarity"], cfg["teeth_white"],
            cfg["blush"], cfg["glow"], cfg["bokeh"],
            cfg["shadow"], cfg["highlight"], cfg["brightness"], cfg["contrast"],
            cfg["saturation"], cfg["sharpness"], cfg["temp"],
            cfg["grain"], cfg["vignette"], cfg["filter"],
            current_rot, current_flip
        )
        
    batch_res = None
    if batch_sample_img is not None:
        batch_res = process_full_studio(
            batch_sample_img,
            cfg["bg_mode"], cfg["bg_feather"],
            cfg["art_style"], cfg["art_blend"], cfg["line_strength"], cfg["cel_shading"],
            cfg["smooth"], cfg["matte"], cfg["detail"], cfg["eye_clarity"], cfg["teeth_white"],
            cfg["blush"], cfg["glow"], cfg["bokeh"],
            cfg["shadow"], cfg["highlight"], cfg["brightness"], cfg["contrast"],
            cfg["saturation"], cfg["sharpness"], cfg["temp"],
            cfg["grain"], cfg["vignette"], cfg["filter"],
            current_rot, current_flip
        )

    return (
        gr.update(value=cfg["bg_mode"]),
        gr.update(value=cfg["bg_feather"]),
        gr.update(value=cfg["art_style"]),
        gr.update(value=cfg["art_blend"]),
        gr.update(value=cfg["line_strength"]),
        gr.update(value=cfg["cel_shading"]),
        gr.update(value=cfg["smooth"]),
        gr.update(value=cfg["matte"]),
        gr.update(value=cfg["detail"]),
        gr.update(value=cfg["eye_clarity"]),
        gr.update(value=cfg["teeth_white"]),
        gr.update(value=cfg["blush"]),
        gr.update(value=cfg["glow"]),
        gr.update(value=cfg["bokeh"]),
        gr.update(value=cfg["shadow"]),
        gr.update(value=cfg["highlight"]),
        gr.update(value=cfg["brightness"]),
        gr.update(value=cfg["contrast"]),
        gr.update(value=cfg["saturation"]),
        gr.update(value=cfg["sharpness"]),
        gr.update(value=cfg["temp"]),
        gr.update(value=cfg["grain"]),
        gr.update(value=cfg["vignette"]),
        gr.update(value=cfg["filter"]),
        single_res if single_res is not None else gr.update(),
        batch_res if batch_res is not None else gr.update()
    )

def apply_universal_adjustments(single_img, batch_sample_img, *params):
    single_res = None
    if single_img is not None:
        single_res = process_full_studio(single_img, *params)
        
    batch_res = None
    if batch_sample_img is not None:
        batch_res = process_full_studio(batch_sample_img, *params)
        
    return (
        single_res if single_res is not None else gr.update(),
        batch_res if batch_res is not None else gr.update()
    )

def reset_all_sliders_and_preview(single_img, batch_sample_img):
    return (
        gr.update(value=DEFAULTS["bg_mode"]),
        gr.update(value=DEFAULTS["bg_feather"]),
        gr.update(value=DEFAULTS["art_style"]),
        gr.update(value=DEFAULTS["art_blend"]),
        gr.update(value=DEFAULTS["line_strength"]),
        gr.update(value=DEFAULTS["cel_shading"]),
        gr.update(value=DEFAULTS["smooth"]),
        gr.update(value=DEFAULTS["matte"]),
        gr.update(value=DEFAULTS["detail"]),
        gr.update(value=DEFAULTS["eye_clarity"]),
        gr.update(value=DEFAULTS["teeth_white"]),
        gr.update(value=DEFAULTS["blush"]),
        gr.update(value=DEFAULTS["glow"]),
        gr.update(value=DEFAULTS["bokeh"]),
        gr.update(value=DEFAULTS["shadow"]),
        gr.update(value=DEFAULTS["highlight"]),
        gr.update(value=DEFAULTS["brightness"]),
        gr.update(value=DEFAULTS["contrast"]),
        gr.update(value=DEFAULTS["saturation"]),
        gr.update(value=DEFAULTS["sharpness"]),
        gr.update(value=DEFAULTS["temp"]),
        gr.update(value=DEFAULTS["grain"]),
        gr.update(value=DEFAULTS["vignette"]),
        gr.update(value=DEFAULTS["filter"]),
        gr.update(value=DEFAULTS["rotation"]),
        gr.update(value=DEFAULTS["flip_h"]),
        single_img if single_img is not None else gr.update(),
        batch_sample_img if batch_sample_img is not None else gr.update()
    )

CUSTOM_CSS = """
/* 隱藏 gr.File 上傳後的長清單列表 */
#batch_file_uploader .file-preview,
#batch_file_uploader table,
#batch_file_uploader .file-preview-holder {
    display: none !important;
}
#batch_file_uploader {
    min-height: 50px !important;
    margin-bottom: 8px !important;
}
/* 徹底隱藏 Gallery 內建大圖預覽區塊 */
#batch_gallery .preview, 
#batch_gallery .main,
#batch_gallery [data-testid="detailed-view"] {
    display: none !important;
    height: 0px !important;
    min-height: 0px !important;
    margin: 0px !important;
    padding: 0px !important;
}
/* 強制將 Gallery 高度貼合縮圖內容，消除下方大片空白 */
#batch_gallery {
    height: auto !important;
    min-height: unset !important;
    max-height: fit-content !important;
    padding-bottom: 5px !important;
}
#batch_gallery .grid-wrap {
    height: auto !important;
    min-height: unset !important;
}
#batch_gallery .thumbnails {
    margin-top: 5px !important;
    gap: 6px !important;
}
#batch_gallery button {
    border-radius: 8px !important;
}
"""

with gr.Blocks(title="自然人像與藝術修圖工作站", css=CUSTOM_CSS) as interface:
    gr.Markdown("## 📷 自然人像與藝術風格修圖工作站 (旗艦全能終極版)")
    gr.Markdown("純記憶體極速運算，無多餘暫存檔。支援 **單圖精修** 與 **點選縮圖切換範本批次打包** 模式。")

    with gr.Tabs():
        # --- 單圖模式分頁 ---
        with gr.TabItem("🖼️ 單圖即時精修"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="🖼️ 原始相片", interactive=True)
                with gr.Column(scale=1):
                    img_output = gr.Image(type="pil", label="✨ 精修成果 (預覽)", format="png", interactive=False)
                    with gr.Row():
                        btn_dl_png = gr.Button("💾 產生高畫質 PNG 下載檔", variant="secondary")
                        btn_dl_jpg = gr.Button("💾 產生高畫質 JPG 下載檔", variant="secondary")
                    file_download = gr.File(label="📥 點擊下方檔案儲存到電腦", visible=False)

        # --- 批次處理模式分頁 ---
        with gr.TabItem("🗂️ 批次多圖處理 (點選縮圖切換範本 ➡️ 滿意再批次打包)"):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_inputs = gr.File(
                        label="📁 選擇或拖曳多張圖片上傳", 
                        file_count="multiple", 
                        file_types=["image"],
                        elem_id="batch_file_uploader"
                    )
                    batch_sample_input = gr.Image(
                        type="pil", 
                        label="🖼️ 當前選定範本照片 (原圖)", 
                        interactive=False
                    )
                    batch_gallery = gr.Gallery(
                        label="點擊縮圖切換範本", 
                        show_label=True, 
                        elem_id="batch_gallery", 
                        columns=6, 
                        rows=2, 
                        height="auto",
                        allow_preview=False
                    )
                    
                with gr.Column(scale=1):
                    batch_sample_output = gr.Image(
                        type="pil", 
                        label="✨ 範本修圖成果預覽 (調整下方參數即時聯動)", 
                        format="png", 
                        interactive=False
                    )
                    batch_run_btn = gr.Button("⚡ 確認效果符合需求，套用至全部照片並打包 ZIP", variant="primary")
                    batch_status = gr.Textbox(label="📊 批次處理進度狀態", value="等待上傳圖片中...", interactive=False)
                    batch_download = gr.File(label="📦 點擊下載批次處理 ZIP 壓縮包", visible=False)

    # 🌟 全域操作動作列
    with gr.Row():
        apply_btn = gr.Button("🚀 套用自訂調整 (更新單圖 ＆ 範本圖)", variant="primary", scale=3)
        reset_btn = gr.Button("🔄 重設數值 (還原為原圖)", variant="secondary", scale=1)

    # ⚡ 一鍵熱門風格 (共 24 款)
    gr.Markdown("### ⚡ 一鍵熱門風格 (經典人像 ＆ 藝術繪畫)")
    preset_buttons = {}
    preset_names = [
        "🌸 膠原蛋白裸妝", "☀️ 自然暖色調", "☁️ 柔和對比", "🌟 雜誌高訂啞光",
        "✨ 韓系清透冷白", "🌿 日系清新空氣感", "☕ 復古法式暖咖", "🧸 復古微醺暖調",
        "🍷 復古法式紅酒", "🏙️ 俐落冷萃都會", "🌊 冰島冷萃清冽", "🎬 電影青橙色調",
        "🖤 黑白經典紀實", "💡 逆光人像救援", "🎨 經典卡通風 (Cartoon)", "🌸 日系動漫風 (Anime)",
        "🖌️ 印象派油畫 (Oil Painting)", "🎨 夢幻水彩畫 (Watercolor)", "🏮 單色墨水渲染 (Ink Wash Sketch)", "🏮 東方水墨風 (Ink Wash)",
        "✏️ 復古素描風 (Pencil Sketch)", "🌆 賽博龐克霓虹 (Cyberpunk)", "👾 復古像素藝術 (Pixel Art)", "🎨 普普藝術風 (Pop Art)"
    ]

    for i in range(0, len(preset_names), 4):
        with gr.Row():
            for name in preset_names[i:i+4]:
                preset_buttons[name] = gr.Button(name, variant="secondary")

    # 參數設定分頁
    with gr.Tabs():
        with gr.TabItem("✂️ AI 髮絲級智慧去背"):
            with gr.Row():
                bg_dropdown = gr.Dropdown(
                    choices=[
                        "保留原圖背景",
                        "✂️ 透明背景 (PNG)",
                        "⚪ 純白證件照背景",
                        "🔵 商務證件藍背景",
                        "🔴 喜慶證件紅背景",
                        "🔘 質感冷灰背景",
                        "🍵 莫蘭迪綠背景",
                        "🖤 背景黑白 (人物全彩聚焦)",
                        "📸 背景大光圈深層虛化 (Bokeh)"
                    ],
                    value=DEFAULTS["bg_mode"],
                    label="背景模式選擇"
                )
                bg_feather_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["bg_feather"], step=0.05, label="去背邊緣容差調節 (預設 0.5 最佳)")

        with gr.TabItem("🎨 藝術繪畫風格"):
            with gr.Row():
                art_dropdown = gr.Dropdown(
                    choices=ART_STYLE_CHOICES,
                    value=DEFAULTS["art_style"],
                    label="藝術風格選擇"
                )
                art_blend_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["art_blend"], step=0.05, label="藝術渲染融合強度")
            with gr.Row():
                line_strength_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["line_strength"], step=0.05, label="輪廓描邊強度 (卡通/動漫/單色水墨線稿/普普風)")
                cel_shading_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["cel_shading"], step=0.05, label="賽璐璐色階簡化度 (色塊層次/水墨擴散感)")

        with gr.TabItem("✨ 人像五官與膚質精修"):
            with gr.Row():
                smooth_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["smooth"], step=0.05, label="自然磨皮 (平滑膚質 / 死守輪廓邊緣)")
                matte_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["matte"], step=0.05, label="面部消油光 (啞光高級感)")
            with gr.Row():
                teeth_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["teeth_white"], step=0.05, label="智慧牙齒美白 (去黃提亮)")
                eye_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["eye_clarity"], step=0.05, label="眼神晶透光 (提升微對比)")
            with gr.Row():
                detail_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["detail"], step=0.05, label="五官立體層次度 (CLAHE)")
                blush_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["blush"], step=0.05, label="好氣色紅潤 (微幅血色感)")

        with gr.TabItem("☀️ 動態光影與曝光"):
            with gr.Row():
                shadow_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["shadow"], step=0.05, label="暗部補光 (挽救背光暗沉)")
                highlight_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["highlight"], step=0.05, label="高光壓制 (修復過曝死白)")
            with gr.Row():
                brightness_slider = gr.Slider(0.5, 1.5, value=DEFAULTS["brightness"], step=0.05, label="整體亮度")
                contrast_slider = gr.Slider(0.5, 1.5, value=DEFAULTS["contrast"], step=0.05, label="對比度")

        with gr.TabItem("🎨 色彩氛圍與清晰度"):
            with gr.Row():
                saturation_slider = gr.Slider(0.0, 2.0, value=DEFAULTS["saturation"], step=0.05, label="色彩飽和度")
                sharpness_slider = gr.Slider(0.0, 2.0, value=DEFAULTS["sharpness"], step=0.05, label="清晰銳利度")
            with gr.Row():
                temp_slider = gr.Slider(-1.0, 1.0, value=DEFAULTS["temp"], step=0.1, label="色溫微調 (冷色調 ↔ 暖色調)")

        with gr.TabItem("🎞️ 底片調色與光學質感"):
            with gr.Row():
                filter_dropdown = gr.Dropdown(
                    choices=[
                        "經典無濾鏡",
                        "🎞️ 富士正片膠卷 (Fuji Film)",
                        "🌅 柯達金感暖調 (Kodak Gold)",
                        "🎬 電影青橙 (Teal & Orange)",
                        "🍷 法式紅酒 (Bordeaux Mood)",
                        "🌊 冷萃青藍 (Nordic Cool)",
                        "🖤 高對比黑白藝術 (Monochrome)"
                    ],
                    value=DEFAULTS["filter"],
                    label="經典膠卷色調風格"
                )
            with gr.Row():
                glow_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["glow"], step=0.05, label="柔焦奶油肌光暈 (Orton Glow)")
                bokeh_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["bokeh"], step=0.05, label="大光圈背景虛化 (DSLR Bokeh)")
            with gr.Row():
                grain_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["grain"], step=0.05, label="復古膠卷顆粒感 (Film Grain)")
                vignette_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["vignette"], step=0.05, label="暗角聚焦效果 (Vignette)")

        with gr.TabItem("📐 構圖校正與翻轉"):
            with gr.Row():
                rotation_dropdown = gr.Dropdown(
                    choices=["0°", "90° 順時針", "180° 翻轉", "270° 逆時針"],
                    value=DEFAULTS["rotation"],
                    label="旋轉角度"
                )
                flip_checkbox = gr.Checkbox(value=DEFAULTS["flip_h"], label="↔️ 水平鏡像翻轉")

    # 元件清單打包
    all_param_controls = [
        bg_dropdown, bg_feather_slider,
        art_dropdown, art_blend_slider, line_strength_slider, cel_shading_slider,
        smooth_slider, matte_slider, detail_slider, eye_slider, teeth_slider, blush_slider, glow_slider, bokeh_slider,
        shadow_slider, highlight_slider, brightness_slider, contrast_slider,
        saturation_slider, sharpness_slider, temp_slider,
        grain_slider, vignette_slider, filter_dropdown
    ]
    all_controls_full = all_param_controls + [rotation_dropdown, flip_checkbox]
    preset_outputs = all_param_controls + [img_output, batch_sample_output]

    # 單圖上傳事件 (上傳後維持原圖)
    img_input.upload(
        fn=lambda img: (img, gr.update(value=DEFAULTS["bg_mode"])),
        inputs=[img_input],
        outputs=[img_output, bg_dropdown]
    )

    # 一鍵風格按鈕綁定 (全面雙向響應)
    for name, btn in preset_buttons.items():
        btn.click(
            fn=lambda s_img, b_img, r, f, p_name=name: load_preset_universal(p_name, s_img, b_img, r, f),
            inputs=[img_input, batch_sample_input, rotation_dropdown, flip_checkbox],
            outputs=preset_outputs
        )

    # 全域「套用自訂調整」按鈕
    apply_btn.click(
        fn=apply_universal_adjustments,
        inputs=[img_input, batch_sample_input] + all_controls_full,
        outputs=[img_output, batch_sample_output]
    )

    btn_dl_png.click(fn=lambda img: export_file_on_demand(img, "png"), inputs=[img_output], outputs=[file_download])
    btn_dl_jpg.click(fn=lambda img: export_file_on_demand(img, "jpg"), inputs=[img_output], outputs=[file_download])

    # 批次上傳事件 (載入純縮圖清單，並預設第 1 張為範本)
    batch_inputs.upload(
        fn=on_batch_files_uploaded,
        inputs=[batch_inputs],
        outputs=[batch_sample_input, batch_sample_output, batch_gallery, batch_status, batch_download]
    )

    # 點擊縮圖切換範本事件
    batch_gallery.select(
        fn=on_gallery_select,
        inputs=[batch_inputs] + all_controls_full,
        outputs=[batch_sample_input, batch_sample_output, batch_status]
    )

    # 批次打包下載
    batch_run_btn.click(
        fn=run_batch_export,
        inputs=[batch_inputs] + all_controls_full,
        outputs=[batch_download, batch_status]
    )

    # 重設數值：全部滑桿歸零，預覽框直接還原為原圖
    reset_btn.click(
        fn=reset_all_sliders_and_preview,
        inputs=[img_input, batch_sample_input],
        outputs=all_controls_full + [img_output, batch_sample_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(),
        allowed_paths=[str(OUTPUT_DIR)]
    )