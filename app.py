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

# --- AI 去背模組 ---
def find_u2net_model():
    search_dirs = [BASE_DIR, Path.cwd(), Path.home() / ".u2net"]
    for d in search_dirs:
        target = d / "u2net.onnx"
        if target.is_file():
            return str(target)
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
        return Image.fromarray(np.dstack([proc_rgb, (alpha * 255).astype(np.uint8)]), "RGBA")

    bg_colors = {
        "⚪ 純白證件照背景": np.array([255, 255, 255], dtype=np.uint8),
        "🔵 商務證件藍背景": np.array([67, 142, 219], dtype=np.uint8),
        "🔴 喜慶證件紅背景": np.array([218, 41, 28], dtype=np.uint8),
        "🔘 質感冷灰背景": np.array([220, 222, 225], dtype=np.uint8),
        "🍵 莫蘭迪綠背景": np.array([178, 190, 181], dtype=np.uint8)
    }

    if bg_mode in bg_colors:
        composed = (proc_rgb * alpha_3d + np.full_like(proc_rgb, bg_colors[bg_mode]) * (1.0 - alpha_3d)).astype(np.uint8)
        return Image.fromarray(composed, "RGB")
    elif bg_mode == "🖤 背景黑白 (人物全彩聚焦)":
        gray_bg = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        return Image.fromarray((proc_rgb * alpha_3d + gray_bg * (1.0 - alpha_3d)).astype(np.uint8), "RGB")
    elif bg_mode == "📸 背景大光圈深層虛化 (Bokeh)":
        blurred_bg = cv2.GaussianBlur(img_rgb, (51, 51), 0)
        return Image.fromarray((proc_rgb * alpha_3d + blurred_bg * (1.0 - alpha_3d)).astype(np.uint8), "RGB")
    return processed_pil

ART_STYLE_CHOICES = [
    "原圖無藝術濾鏡", "🎨 經典卡通風 (Cartoon)", "🌸 日系動漫風 (Anime)",
    "🖌️ 印象派油畫 (Oil Painting)", "🎨 夢幻水彩畫 (Watercolor)", "🏮 單色墨水渲染 (Ink Wash Sketch)",
    "🏮 東方水墨風 (Ink Wash)", "✏️ 復古素描風 (Pencil Sketch)", "🌆 賽博龐克霓虹 (Cyberpunk)",
    "👾 復古像素藝術 (Pixel Art)", "🎨 普普藝術風 (Pop Art)"
]

DEFAULTS = {
    "bg_mode": "保留原圖背景", "bg_feather": 0.5, "art_style": "原圖無藝術濾鏡",
    "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5, "smooth": 0.0,
    "matte": 0.0, "detail": 0.0, "eye_clarity": 0.0, "teeth_white": 0.0,
    "blush": 0.0, "glow": 0.0, "bokeh": 0.0, "shadow": 0.0, "highlight": 0.0,
    "brightness": 1.00, "contrast": 1.00, "saturation": 1.00, "sharpness": 1.00,
    "temp": 0.0, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡",
    "rotation": "0°", "flip_h": False
}

PRESETS = {
    "🌸 膠原蛋白裸妝": {"bg_mode": "保留原圖背景", "bg_feather": 0.5, "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5, "smooth": 0.15, "matte": 0.0, "detail": 0.05, "eye_clarity": 0.10, "teeth_white": 0.0, "blush": 0.08, "glow": 0.10, "bokeh": 0.0, "shadow": 0.05, "highlight": 0.05, "brightness": 1.03, "contrast": 1.01, "saturation": 1.02, "sharpness": 1.02, "temp": 0.02, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"},
    "☀️ 自然暖色調": {"bg_mode": "保留原圖背景", "bg_feather": 0.5, "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5, "smooth": 0.15, "matte": 0.0, "detail": 0.05, "eye_clarity": 0.10, "teeth_white": 0.05, "blush": 0.10, "glow": 0.08, "bokeh": 0.0, "shadow": 0.05, "highlight": 0.05, "brightness": 1.03, "contrast": 1.02, "saturation": 1.05, "sharpness": 1.02, "temp": 0.18, "grain": 0.0, "vignette": 0.05, "filter": "經典無濾鏡"},
    "✨ 韓系清透冷白": {"bg_mode": "保留原圖背景", "bg_feather": 0.5, "art_style": "原圖無藝術濾鏡", "art_blend": 1.0, "line_strength": 0.5, "cel_shading": 0.5, "smooth": 0.20, "matte": 0.10, "detail": 0.05, "eye_clarity": 0.15, "teeth_white": 0.20, "blush": 0.05, "glow": 0.15, "bokeh": 0.05, "shadow": 0.10, "highlight": 0.05, "brightness": 1.05, "contrast": 0.99, "saturation": 0.98, "sharpness": 1.02, "temp": -0.10, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"},
    "🎨 經典卡通風 (Cartoon)": {"bg_mode": "保留原圖背景", "bg_feather": 0.5, "art_style": "🎨 經典卡通風 (Cartoon)", "art_blend": 0.85, "line_strength": 0.60, "cel_shading": 0.60, "smooth": 0.10, "matte": 0.0, "detail": 0.10, "eye_clarity": 0.10, "teeth_white": 0.0, "blush": 0.0, "glow": 0.0, "bokeh": 0.0, "shadow": 0.0, "highlight": 0.0, "brightness": 1.05, "contrast": 1.10, "saturation": 1.15, "sharpness": 1.10, "temp": 0.0, "grain": 0.0, "vignette": 0.0, "filter": "經典無濾鏡"}
}

def render_art_style(img_bgr, style_name, blend_ratio, line_strength, cel_shading):
    if style_name == "原圖無藝術濾鏡" or blend_ratio <= 0:
        return img_bgr
    art_bgr = img_bgr.copy()
    if style_name == "🎨 經典卡通風 (Cartoon)":
        smooth = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=90, sigmaSpace=90)
        gray = cv2.medianBlur(cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY), 5)
        edges = cv2.cvtColor(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2 + int(line_strength * 5)), cv2.COLOR_GRAY2BGR)
        div = max(int(40 + (1.0 - cel_shading) * 40), 20)
        art_bgr = cv2.bitwise_and((smooth // div) * div + div // 2, edges)
    return cv2.addWeighted(art_bgr, blend_ratio, img_bgr, 1.0 - blend_ratio, 0)

def process_full_studio(input_img, bg_mode, bg_feather, art_style, art_blend, line_s, cel_s, smooth, matte, detail, eye, teeth, blush, glow, bokeh, shadow, highlight, bright, contrast, sat, sharp, temp, grain, vignette, filt, rot, flip):
    if input_img is None:
        return None
    img_bgr = cv2.cvtColor(np.array(input_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    if rot == "90° 順時針": img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rot == "180° 翻轉": img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)
    elif rot == "270° 逆時針": img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip: img_bgr = cv2.flip(img_bgr, 1)

    img_bgr = render_art_style(img_bgr, art_style, art_blend, line_s, cel_s)
    if smooth > 0:
        smooth_bgr = cv2.bilateralFilter(img_bgr, d=int(smooth * 10) + 1, sigmaColor=smooth * 40, sigmaSpace=smooth * 40)
        img_bgr = cv2.addWeighted(smooth_bgr, smooth * 0.5, img_bgr, 1 - (smooth * 0.5), 0)

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if bright != 1.0: img_pil = ImageEnhance.Brightness(img_pil).enhance(bright)
    if contrast != 1.0: img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
    if sat != 1.0: img_pil = ImageEnhance.Color(img_pil).enhance(sat)
    if sharp != 1.0: img_pil = ImageEnhance.Sharpness(img_pil).enhance(sharp)
    return apply_background_matting(img_pil, input_img, bg_mode, bg_feather)

def export_file_on_demand(current_img, file_format):
    if current_img is None: return None
    timestamp = int(time.time())
    ext = "png" if (getattr(current_img, "mode", "RGB") == "RGBA" or file_format.lower() == "png") else "jpg"
    save_path = str(OUTPUT_DIR / f"enhanced_{timestamp}.{ext}")
    if ext == "png": current_img.save(save_path, format="PNG")
    else: current_img.convert("RGB").save(save_path, format="JPEG", quality=95)
    return gr.update(value=save_path, visible=True)

def on_batch_uploaded(file_list):
    if not file_list: return None, None, [], "請先上傳圖片！", gr.update(visible=False)
    items = [(f.name if hasattr(f, "name") else str(f), f"第 {i+1} 張") for i, f in enumerate(file_list)]
    first_pil = Image.open(items[0][0]).convert("RGB")
    return first_pil, first_pil, items, f"📂 已成功載入 {len(file_list)} 張照片", gr.update(visible=False)

def on_gallery_select(evt: gr.SelectData, file_list, *params):
    if not file_list: return gr.update(), gr.update(), "無可選取的圖片"
    path = file_list[evt.index].name if hasattr(file_list[evt.index], "name") else str(file_list[evt.index])
    sel_pil = Image.open(path).convert("RGB")
    return sel_pil, process_full_studio(sel_pil, *params), f"🎯 已切換範本為第 {evt.index+1} 張！"

def run_batch_export(file_list, *params):
    if not file_list: return gr.update(visible=False), "請先上傳圖片！"
    zip_path = str(OUTPUT_DIR / f"batch_{int(time.time())}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in file_list:
            src = f.name if hasattr(f, "name") else str(f)
            res = process_full_studio(Image.open(src).convert("RGB"), *params)
            buf = io.BytesIO()
            ext = ".png" if getattr(res, "mode", "RGB") == "RGBA" else ".jpg"
            if ext == ".png": res.save(buf, format="PNG")
            else: res.convert("RGB").save(buf, format="JPEG", quality=95)
            zipf.writestr(f"{Path(src).stem}_enhanced{ext}", buf.getvalue())
    return gr.update(value=zip_path, visible=True), "🎉 批次打包完成！"

CUSTOM_CSS = """
#batch_file_uploader .file-preview, #batch_file_uploader table, #batch_file_uploader .file-preview-holder { display: none !important; }
#batch_file_uploader { min-height: 50px !important; margin-bottom: 8px !important; }
#batch_gallery .preview, #batch_gallery .main, #batch_gallery [data-testid="detailed-view"] { display: none !important; height: 0px !important; }
#batch_gallery { height: auto !important; max-height: fit-content !important; padding-bottom: 5px !important; }
#batch_gallery .grid-wrap { height: auto !important; }
#batch_gallery .thumbnails { margin-top: 5px !important; gap: 6px !important; }
#batch_gallery button { border-radius: 8px !important; }
"""

with gr.Blocks(title="自然人像與藝術修圖工作站", css=CUSTOM_CSS) as interface:
    gr.Markdown("## 📷 自然人像與藝術風格修圖工作站 (雲端版)")
    with gr.Tabs():
        with gr.TabItem("🖼️ 單圖即時精修"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="原始相片", interactive=True)
                with gr.Column():
                    img_output = gr.Image(type="pil", label="精修成果", format="png", interactive=False)
                    with gr.Row():
                        btn_dl_png = gr.Button("💾 下載 PNG", variant="secondary")
                        btn_dl_jpg = gr.Button("💾 下載 JPG", variant="secondary")
                    file_download = gr.File(label="下載檔案", visible=False)

        with gr.TabItem("🗂️ 批次多圖處理"):
            with gr.Row():
                with gr.Column():
                    batch_inputs = gr.File(label="選擇多張圖片", file_count="multiple", file_types=["image"], elem_id="batch_file_uploader")
                    batch_sample_input = gr.Image(type="pil", label="選定範本照片", interactive=False)
                    batch_gallery = gr.Gallery(label="點擊縮圖切換範本", elem_id="batch_gallery", columns=6, rows=2, height="auto", allow_preview=False)
                with gr.Column():
                    batch_sample_output = gr.Image(type="pil", label="成果預覽", format="png", interactive=False)
                    batch_run_btn = gr.Button("⚡ 套用全部並打包 ZIP", variant="primary")
                    batch_status = gr.Textbox(label="處理進度", value="等待上傳...", interactive=False)
                    batch_download = gr.File(label="下載 ZIP", visible=False)

    with gr.Row():
        apply_btn = gr.Button("🚀 套用自訂調整", variant="primary", scale=3)
        reset_btn = gr.Button("🔄 重設數值", variant="secondary", scale=1)

    with gr.Tabs():
        with gr.TabItem("✂️ AI 去背"):
            with gr.Row():
                bg_dropdown = gr.Dropdown(choices=["保留原圖背景", "✂️ 透明背景 (PNG)", "⚪ 純白證件照背景", "🔵 商務證件藍背景", "🔴 喜慶證件紅背景", "🔘 質感冷灰背景", "🍵 莫蘭迪綠背景", "🖤 背景黑白 (人物全彩聚焦)", "📸 背景大光圈深層虛化 (Bokeh)"], value=DEFAULTS["bg_mode"], label="背景模式")
                bg_feather_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["bg_feather"], step=0.05, label="邊緣容差")
        with gr.TabItem("🎨 藝術風格"):
            with gr.Row():
                art_dropdown = gr.Dropdown(choices=ART_STYLE_CHOICES, value=DEFAULTS["art_style"], label="風格選擇")
                art_blend_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["art_blend"], step=0.05, label="融合強度")
            with gr.Row():
                line_strength_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["line_strength"], step=0.05, label="描邊強度")
                cel_shading_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["cel_shading"], step=0.05, label="色階簡化")
        with gr.TabItem("✨ 人像精修"):
            with gr.Row():
                smooth_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["smooth"], step=0.05, label="自然磨皮")
                matte_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["matte"], step=0.05, label="消油光")
            with gr.Row():
                teeth_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["teeth_white"], step=0.05, label="牙齒美白")
                eye_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["eye_clarity"], step=0.05, label="眼神光")
                detail_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["detail"], step=0.05, label="立體層次")
                blush_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["blush"], step=0.05, label="好氣色")
        with gr.TabItem("☀️ 光影與調色"):
            with gr.Row():
                shadow_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["shadow"], step=0.05, label="暗部補光")
                highlight_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["highlight"], step=0.05, label="高光壓制")
                brightness_slider = gr.Slider(0.5, 1.5, value=DEFAULTS["brightness"], step=0.05, label="整體亮度")
                contrast_slider = gr.Slider(0.5, 1.5, value=DEFAULTS["contrast"], step=0.05, label="對比度")
            with gr.Row():
                saturation_slider = gr.Slider(0.0, 2.0, value=DEFAULTS["saturation"], step=0.05, label="飽和度")
                sharpness_slider = gr.Slider(0.0, 2.0, value=DEFAULTS["sharpness"], step=0.05, label="銳利度")
                temp_slider = gr.Slider(-1.0, 1.0, value=DEFAULTS["temp"], step=0.1, label="色溫")
        with gr.TabItem("🎞️ 底片濾鏡"):
            filter_dropdown = gr.Dropdown(choices=["經典無濾鏡", "🎞️ 富士正片膠卷 (Fuji Film)", "🌅 柯達金感暖調 (Kodak Gold)", "🎬 電影青橙 (Teal & Orange)", "🍷 法式紅酒 (Bordeaux Mood)", "🌊 冷萃青藍 (Nordic Cool)", "🖤 高對比黑白藝術 (Monochrome)"], value=DEFAULTS["filter"], label="濾鏡選擇")
            with gr.Row():
                glow_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["glow"], step=0.05, label="光暈")
                bokeh_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["bokeh"], step=0.05, label="背景虛化")
                grain_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["grain"], step=0.05, label="顆粒")
                vignette_slider = gr.Slider(0.0, 1.0, value=DEFAULTS["vignette"], step=0.05, label="暗角")
        with gr.TabItem("📐 構圖"):
            with gr.Row():
                rotation_dropdown = gr.Dropdown(choices=["0°", "90° 順時針", "180° 翻轉", "270° 逆時針"], value=DEFAULTS["rotation"], label="旋轉")
                flip_checkbox = gr.Checkbox(value=DEFAULTS["flip_h"], label="水平翻轉")

    all_controls = [bg_dropdown, bg_feather_slider, art_dropdown, art_blend_slider, line_strength_slider, cel_shading_slider, smooth_slider, matte_slider, detail_slider, eye_slider, teeth_slider, blush_slider, glow_slider, bokeh_slider, shadow_slider, highlight_slider, brightness_slider, contrast_slider, saturation_slider, sharpness_slider, temp_slider, grain_slider, vignette_slider, filter_dropdown, rotation_dropdown, flip_checkbox]

    img_input.upload(fn=lambda img: (img, gr.update(value=DEFAULTS["bg_mode"])), inputs=[img_input], outputs=[img_output, bg_dropdown])
    apply_btn.click(fn=lambda s, b, *p: (process_full_studio(s, *p) if s else gr.update(), process_full_studio(b, *p) if b else gr.update()), inputs=[img_input, batch_sample_input] + all_controls, outputs=[img_output, batch_sample_output])
    btn_dl_png.click(fn=lambda img: export_file_on_demand(img, "png"), inputs=[img_output], outputs=[file_download])
    btn_dl_jpg.click(fn=lambda img: export_file_on_demand(img, "jpg"), inputs=[img_output], outputs=[file_download])
    batch_inputs.upload(fn=on_batch_uploaded, inputs=[batch_inputs], outputs=[batch_sample_input, batch_sample_output, batch_gallery, batch_status, batch_download])
    batch_gallery.select(fn=on_gallery_select, inputs=[batch_inputs] + all_controls, outputs=[batch_sample_input, batch_sample_output, batch_status])
    batch_run_btn.click(fn=run_batch_export, inputs=[batch_inputs] + all_controls, outputs=[batch_download, batch_status])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    interface.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())