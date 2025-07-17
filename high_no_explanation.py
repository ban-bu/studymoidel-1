import streamlit as st
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import os  # 确保os模块在这里导入
# 移除cairosvg依赖，使用svglib作为唯一的SVG处理库
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    SVGLIB_AVAILABLE = True
except ImportError:
    SVGLIB_AVAILABLE = False
    st.warning("SVG processing libraries not installed, SVG conversion will not be available")
from openai import OpenAI
from streamlit_image_coordinates import streamlit_image_coordinates
import re
import math
# 导入面料纹理模块
from fabric_texture import apply_fabric_texture
import uuid
import json
# 导入并行处理库
import concurrent.futures
import time
import threading
# 导入阿里云DashScope文生图API
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
try:
    from dashscope import ImageSynthesis
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    st.warning("DashScope not installed, will use OpenAI DALL-E as fallback")

# API配置信息 - 多个API密钥用于增强并发能力
API_KEYS = [
    "sk-lNVAREVHjj386FDCd9McOL7k66DZCUkTp6IbV0u9970qqdlg",
    "sk-y8x6LH0zdtyQncT0aYdUW7eJZ7v7cuKTp90L7TiK3rPu3fAg", 
    "sk-Kp59pIj8PfqzLzYaAABh2jKsQLB0cUKU3n8l7TIK3rpU61QG",
    "sk-KACPocnavR6poutXUaj7HxsqUrxvcV808S2bv0U9974Ec83g",
    "sk-YknuN0pb6fKBOP6xFOqAdeeqhoYkd1cEl9380vC5HHeC2B30"
]
BASE_URL = "https://api.deepbricks.ai/v1/"

# GPT-4o-mini API配置 - 同样使用多个密钥
GPT4O_MINI_API_KEYS = [
    "sk-lNVAREVHjj386FDCd9McOL7k66DZCUkTp6IbV0u9970qqdlg",
    "sk-y8x6LH0zdtyQncT0aYdUW7eJZ7v7cuKTp90L7TiK3rPu3fAg",
    "sk-Kp59pIj8PfqzLzYaAABh2jKsQLB0cUKU3n8l7TIK3rpU61QG", 
    "sk-KACPocnavR6poutXUaj7HxsqUrxvcV808S2bv0U9974Ec83g",
    "sk-YknuN0pb6fKBOP6xFOqAdeeqhoYkd1cEl9380vC5HHeC2B30"
]
GPT4O_MINI_BASE_URL = "https://api.deepbricks.ai/v1/"

# 阿里云DashScope API配置
DASHSCOPE_API_KEY = "sk-4f82c6e2097440f8adb2ef688c7c7551"

# API密钥轮询计数器
_api_key_counter = 0
_gpt4o_api_key_counter = 0
_api_lock = threading.Lock()

def get_next_api_key():
    """获取下一个DALL-E API密钥（轮询方式）"""
    global _api_key_counter
    with _api_lock:
        key = API_KEYS[_api_key_counter % len(API_KEYS)]
        _api_key_counter += 1
        return key

def get_next_gpt4o_api_key():
    """获取下一个GPT-4o-mini API密钥（轮询方式）"""
    global _gpt4o_api_key_counter
    with _api_lock:
        key = GPT4O_MINI_API_KEYS[_gpt4o_api_key_counter % len(GPT4O_MINI_API_KEYS)]
        _gpt4o_api_key_counter += 1
        return key

def make_background_transparent(image, threshold=100):
    """
    将图像的白色/浅色背景转换为透明背景
    
    Args:
        image: PIL图像对象，RGBA模式
        threshold: 背景色识别阈值，数值越大识别的背景范围越大
    
    Returns:
        处理后的PIL图像对象，透明背景
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # 获取图像数据
    data = image.getdata()
    new_data = []
    
    # 分析四个角落的颜色来确定背景色
    width, height = image.size
    corner_pixels = [
        image.getpixel((0, 0)),           # 左上角
        image.getpixel((width-1, 0)),     # 右上角
        image.getpixel((0, height-1)),    # 左下角
        image.getpixel((width-1, height-1)) # 右下角
    ]
    
    # 计算平均背景颜色（假设四个角都是背景）
    bg_r = sum(p[0] for p in corner_pixels) // 4
    bg_g = sum(p[1] for p in corner_pixels) // 4
    bg_b = sum(p[2] for p in corner_pixels) // 4
    
    print(f"检测到的背景颜色: RGB({bg_r}, {bg_g}, {bg_b})")
    
    # 遍历所有像素
    transparent_count = 0
    for item in data:
        r, g, b, a = item
        
        # 计算当前像素与背景色的差异
        diff = abs(r - bg_r) + abs(g - bg_g) + abs(b - bg_b)
        
        # 另外检查是否是浅色（可能是背景）
        brightness = (r + g + b) / 3
        is_light = brightness > 180  # 亮度大于180认为是浅色
        
        # 检查是否接近灰白色
        gray_similarity = abs(r - g) + abs(g - b) + abs(r - b)
        is_grayish = gray_similarity < 30  # 颜色差异小说明是灰色系
        
        # 如果差异小于阈值或者是浅色灰白色，认为是背景，设为透明
        if diff < threshold or (is_light and is_grayish):
            new_data.append((r, g, b, 0))  # 完全透明
            transparent_count += 1
        else:
            # 否则保持原像素
            new_data.append((r, g, b, a))
    
    print(f"透明化了 {transparent_count} 个像素，占总像素的 {transparent_count/(image.size[0]*image.size[1])*100:.1f}%")
    
    # 创建新图像
    transparent_image = Image.new('RGBA', image.size)
    transparent_image.putdata(new_data)
    
    return transparent_image

# 自定义SVG转PNG函数，不依赖外部库
def convert_svg_to_png(svg_content):
    """
    将SVG内容转换为PNG格式的PIL图像对象
    使用svglib库来处理，不再依赖cairosvg
    """
    try:
        if SVGLIB_AVAILABLE:
            # 使用svglib将SVG内容转换为PNG
            from io import BytesIO
            svg_bytes = BytesIO(svg_content)
            drawing = svg2rlg(svg_bytes)
            png_bytes = BytesIO()
            renderPM.drawToFile(drawing, png_bytes, fmt="PNG")
            png_bytes.seek(0)
            return Image.open(png_bytes).convert("RGBA")
        else:
            st.error("SVG conversion libraries not available. Please install svglib and reportlab.")
            return None
    except Exception as e:
        st.error(f"Error converting SVG to PNG: {str(e)}")
        return None

# 设置默认生成的设计数量，取代UI上的选择按钮
DEFAULT_DESIGN_COUNT = 1  # 可以设置为1, 3, 5，分别对应原来的low, medium, high

def get_ai_design_suggestions(user_preferences=None):
    """Get design suggestions from GPT-4o-mini with more personalized features"""
    client = OpenAI(api_key=get_next_gpt4o_api_key(), base_url=GPT4O_MINI_BASE_URL)
    
    # Default prompt if no user preferences provided
    if not user_preferences:
        user_preferences = "casual fashion t-shirt design"
    
    # Construct the prompt
    prompt = f"""
    As a design consultant, please provide personalized design suggestions for a "{user_preferences}" style.
    
    Please provide the following design suggestions in JSON format:

    1. Color: Select the most suitable color for this style (provide name and hex code)
    2. Fabric: Select the most suitable fabric type (Cotton, Polyester, Cotton-Polyester Blend, Jersey, Linen, or Bamboo)
    3. Text: A suitable phrase or slogan that matches the style (keep it concise and impactful)
    4. Logo: A brief description of a logo element that would complement the design

    Return your response as a valid JSON object with the following structure:
    {{
        "color": {{
            "name": "Color name",
            "hex": "#XXXXXX"
        }},
        "fabric": "Fabric type",
        "text": "Suggested text or slogan",
        "logo": "Logo/graphic description"
    }}
    """
    
    try:
        # 调用GPT-4o-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional design consultant. Provide design suggestions in JSON format exactly as requested."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # 返回建议内容
        if response.choices and len(response.choices) > 0:
            suggestion_text = response.choices[0].message.content
            
            # 尝试解析JSON
            try:
                # 查找JSON格式的内容
                json_match = re.search(r'```json\s*(.*?)\s*```', suggestion_text, re.DOTALL)
                if json_match:
                    suggestion_json = json.loads(json_match.group(1))
                else:
                    # 尝试直接解析整个内容
                    suggestion_json = json.loads(suggestion_text)
                
                return suggestion_json
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                return {"error": f"Failed to parse design suggestions: {str(e)}"}
        else:
            return {"error": "Failed to get AI design suggestions. Please try again later."}
    except Exception as e:
        return {"error": f"Error getting AI design suggestions: {str(e)}"}

def generate_vector_image(prompt, background_color=None):
    """Generate a vector-style logo with transparent background using DashScope API"""
    
    # 构建矢量图logo专用的提示词
    vector_style_prompt = f"""创建一个矢量风格的logo设计: {prompt}
    要求:
    1. 简洁的矢量图风格，线条清晰
    2. 必须是透明背景，不能有任何白色或彩色背景
    3. 专业的logo设计，适合印刷到T恤上
    4. 高对比度，颜色鲜明
    5. 几何形状简洁，不要过于复杂
    6. 不要包含文字或字母
    7. 不要显示T恤或服装模型
    8. 纯粹的图形标志设计
    9. 矢量插画风格，扁平化设计
    10. 重要：背景必须完全透明，不能有任何颜色填充
    11. 请生成PNG格式的透明背景图标
    12. 图标应该是独立的，没有任何背景元素"""
    

    
    # 优先使用DashScope API
    if DASHSCOPE_AVAILABLE:
        try:
            print(f'----使用DashScope生成矢量logo，提示词: {vector_style_prompt}----')
            rsp = ImageSynthesis.call(
                api_key=DASHSCOPE_API_KEY,
                model="wanx2.0-t2i-turbo",
                prompt=vector_style_prompt,
                n=1,
                size='1024*1024'
            )
            print('DashScope响应: %s' % rsp)
            
            if rsp.status_code == HTTPStatus.OK:
                # 下载生成的图像
                for result in rsp.output.results:
                    image_resp = requests.get(result.url)
                    if image_resp.status_code == 200:
                        # 加载图像并转换为RGBA模式
                        img = Image.open(BytesIO(image_resp.content)).convert("RGBA")
                        print(f"DashScope生成的logo尺寸: {img.size}")
                        
                        # 后处理：将白色背景转换为透明（使用更高的阈值）
                        img_processed = make_background_transparent(img, threshold=120)
                        print(f"背景透明化处理完成")
                        return img_processed
                    else:
                        st.error(f"下载图像失败, 状态码: {image_resp.status_code}")
            else:
                print('DashScope调用失败, status_code: %s, code: %s, message: %s' %
                      (rsp.status_code, rsp.code, rsp.message))
                st.error(f"DashScope API调用失败: {rsp.message}")
                
        except Exception as e:
            st.error(f"DashScope API调用错误: {e}")
            print(f"DashScope错误: {e}")
    
    # 如果DashScope不可用，直接返回None
    if not DASHSCOPE_AVAILABLE:
        st.error("DashScope API不可用，无法生成logo。请确保已正确安装dashscope库。")
        return None
    
    # DashScope失败时也直接返回None，不使用备选方案
    st.error("DashScope API调用失败，无法生成logo。请检查网络连接或API密钥。")
    return None

def change_shirt_color(image, color_hex, apply_texture=False, fabric_type=None):
    """Change T-shirt color with optional fabric texture"""
    # 转换十六进制颜色为RGB
    color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # 创建副本避免修改原图
    colored_image = image.copy().convert("RGBA")
    
    # 获取图像数据
    data = colored_image.getdata()
    
    # 创建新数据
    new_data = []
    # 白色阈值 - 调整这个值可以控制哪些像素被视为白色/浅色并被改变
    threshold = 200
    
    for item in data:
        # 判断是否是白色/浅色区域 (RGB值都很高)
        if item[0] > threshold and item[1] > threshold and item[2] > threshold and item[3] > 0:
            # 保持原透明度，改变颜色
            new_color = (color_rgb[0], color_rgb[1], color_rgb[2], item[3])
            new_data.append(new_color)
        else:
            # 保持其他颜色不变
            new_data.append(item)
    
    # 更新图像数据
    colored_image.putdata(new_data)
    
    # 如果需要应用纹理
    if apply_texture and fabric_type:
        return apply_fabric_texture(colored_image, fabric_type)
    
    return colored_image

def apply_text_to_shirt(image, text, color_hex="#FFFFFF", font_size=80):
    """Apply text to T-shirt image"""
    if not text:
        return image
    
    # 创建副本避免修改原图
    result_image = image.copy().convert("RGBA")
    img_width, img_height = result_image.size
    
    # 创建透明的文本图层
    text_layer = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)
    
    # 尝试加载字体
    from PIL import ImageFont
    import platform
    
    font = None
    try:
        system = platform.system()
        
        # 根据不同系统尝试不同的字体路径
        if system == 'Windows':
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/ARIAL.TTF",
                "C:/Windows/Fonts/calibri.ttf",
            ]
        elif system == 'Darwin':  # macOS
            font_paths = [
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
            ]
        else:  # Linux或其他
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            ]
        
        # 尝试加载每个字体
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
    except Exception as e:
        print(f"Error loading font: {e}")
    
    # 如果加载失败，使用默认字体
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            print("Could not load default font")
            return result_image
    
    # 将十六进制颜色转换为RGB
    color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    text_color = color_rgb + (255,)  # 添加不透明度
    
    # 计算文本位置 (居中)
    text_bbox = text_draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = (img_width - text_width) // 2
    text_y = (img_height // 3) - (text_height // 2)  # 放在T恤上部位置
    
    # 绘制文本
    text_draw.text((text_x, text_y), text, fill=text_color, font=font)
    
    # 组合图像
    result_image = Image.alpha_composite(result_image, text_layer)
    
    return result_image

def apply_logo_to_shirt(shirt_image, logo_image, position="center", size_percent=60, background_color=None):
    """Apply logo to T-shirt image with better blending to reduce shadows"""
    if logo_image is None:
        return shirt_image
    
    # 创建副本避免修改原图
    result_image = shirt_image.copy().convert("RGBA")
    img_width, img_height = result_image.size
    
    # 定义T恤前胸区域
    chest_width = int(img_width * 0.95)
    chest_height = int(img_height * 0.6)
    chest_left = (img_width - chest_width) // 2
    chest_top = int(img_height * 0.2)
    
    # 提取logo前景
    logo_with_bg = logo_image.copy().convert("RGBA")
    
    # 调整Logo大小
    logo_size_factor = size_percent / 100
    logo_width = int(chest_width * logo_size_factor * 0.7)
    logo_height = int(logo_width * logo_with_bg.height / logo_with_bg.width)
    logo_resized = logo_with_bg.resize((logo_width, logo_height), Image.LANCZOS)
    
    # 根据位置确定坐标
    position = position.lower() if isinstance(position, str) else "center"
    
    if position == "top-center":
        logo_x, logo_y = chest_left + (chest_width - logo_width) // 2, chest_top + 10
    elif position == "center":
        logo_x, logo_y = chest_left + (chest_width - logo_width) // 2, chest_top + (chest_height - logo_height) // 2 + 30  # 略微偏下
    else:  # 默认中间
        logo_x, logo_y = chest_left + (chest_width - logo_width) // 2, chest_top + (chest_height - logo_height) // 2 + 30
    
    # 对于透明背景的logo，直接使用alpha通道作为蒙版
    if logo_resized.mode == 'RGBA':
        # 使用alpha通道作为蒙版
        logo_mask = logo_resized.split()[-1]  # 获取alpha通道
        print(f"使用RGBA模式logo的alpha通道作为蒙版")
    else:
        # 如果不是RGBA模式，创建传统的基于颜色差异的蒙版
        logo_mask = Image.new("L", logo_resized.size, 0)  # 创建一个黑色蒙版（透明）
        
        # 如果提供了背景颜色，使用它来判断什么是背景
        if background_color:
            bg_color_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        else:
            # 默认假设白色是背景
            bg_color_rgb = (255, 255, 255)
        
        # 遍历像素，创建蒙版
        for y in range(logo_resized.height):
            for x in range(logo_resized.width):
                pixel = logo_resized.getpixel((x, y))
                if len(pixel) >= 3:  # 至少有RGB值
                    # 计算与背景颜色的差异
                    r_diff = abs(pixel[0] - bg_color_rgb[0])
                    g_diff = abs(pixel[1] - bg_color_rgb[1])
                    b_diff = abs(pixel[2] - bg_color_rgb[2])
                    diff = r_diff + g_diff + b_diff
                    
                    # 如果差异大于阈值，则认为是前景
                    if diff > 60:  # 可以调整阈值
                        # 根据差异程度设置不同的透明度
                        transparency = min(255, diff)
                        logo_mask.putpixel((x, y), transparency)
    
    # 对于透明背景的logo，使用PIL的alpha合成功能
    if logo_resized.mode == 'RGBA':
        # 检查logo是否真的有透明像素
        has_transparency = False
        for pixel in logo_resized.getdata():
            if len(pixel) == 4 and pixel[3] < 255:  # 有alpha通道且不完全不透明
                has_transparency = True
                break
        
        print(f"Logo模式: {logo_resized.mode}, 有透明像素: {has_transparency}")
        
        if has_transparency:
            # 直接使用PIL的alpha合成，这样处理透明背景更准确
            print(f"apply_logo_to_shirt: ({logo_x}, {logo_y})")
            result_image.paste(logo_resized, (logo_x, logo_y), logo_resized)
        else:
            # 如果没有透明像素，先处理背景透明化
            print("apply_logo_to_shirt")
            transparent_logo = make_background_transparent(logo_resized, threshold=120)
            result_image.paste(transparent_logo, (logo_x, logo_y), transparent_logo)
    else:
        # 对于非透明背景的logo，使用传统的像素级混合方法
        shirt_region = result_image.crop((logo_x, logo_y, logo_x + logo_width, logo_y + logo_height))
        
        # 合成logo和T恤区域，使用蒙版确保只有logo的非背景部分被使用
        for y in range(logo_height):
            for x in range(logo_width):
                mask_value = logo_mask.getpixel((x, y))
                if mask_value > 20:  # 有一定的不透明度
                    # 获取logo像素
                    logo_pixel = logo_resized.getpixel((x, y))
                    # 获取T恤对应位置的像素
                    shirt_pixel = shirt_region.getpixel((x, y))
                    
                    # 根据透明度混合像素
                    alpha = mask_value / 255.0
                    blended_pixel = (
                        int(logo_pixel[0] * alpha + shirt_pixel[0] * (1 - alpha)),
                        int(logo_pixel[1] * alpha + shirt_pixel[1] * (1 - alpha)),
                        int(logo_pixel[2] * alpha + shirt_pixel[2] * (1 - alpha)),
                        255  # 完全不透明
                    )
                    
                    # 更新T恤区域的像素
                    shirt_region.putpixel((x, y), blended_pixel)
        
        # 将修改后的区域粘贴回T恤
        result_image.paste(shirt_region, (logo_x, logo_y))
    
    return result_image

def generate_complete_design(design_prompt, variation_id=None):
    """Generate complete T-shirt design based on prompt"""
    if not design_prompt:
        return None, {"error": "Please enter a design prompt"}
    
    # 获取AI设计建议
    design_suggestions = get_ai_design_suggestions(design_prompt)
    
    if "error" in design_suggestions:
        return None, design_suggestions
    
    # 加载原始T恤图像
    try:
        original_image_path = "white_shirt.png"
        possible_paths = [
            "white_shirt.png",
            "./white_shirt.png",
            "../white_shirt.png",
            "images/white_shirt.png",
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                original_image_path = path
                found = True
                break
        
        if not found:
            return None, {"error": "Could not find base T-shirt image"}
        
        # 加载原始白色T恤图像
        original_image = Image.open(original_image_path).convert("RGBA")
    except Exception as e:
        return None, {"error": f"Error loading T-shirt image: {str(e)}"}
    
    try:
        # 使用AI建议的颜色和面料
        color_hex = design_suggestions.get("color", {}).get("hex", "#FFFFFF")
        color_name = design_suggestions.get("color", {}).get("name", "Custom Color")
        fabric_type = design_suggestions.get("fabric", "Cotton")
        
        # 1. 应用颜色和纹理
        colored_shirt = change_shirt_color(
            original_image,
            color_hex,
            apply_texture=True,
            fabric_type=fabric_type
        )
        
        # 2. 生成Logo
        logo_description = design_suggestions.get("logo", "")
        logo_image = None
        
        if logo_description:
            # 修改Logo提示词，生成透明背景的矢量图logo
            logo_prompt = f"""Create a professional vector logo design: {logo_description}. 
            Requirements: 
            1. Simple professional design
            2. IMPORTANT: Transparent background (PNG format)
            3. Clear and distinct graphic with high contrast
            4. Vector-style illustration suitable for T-shirt printing
            5. Must not include any text, numbers or color name, only logo graphic
            6. IMPORTANT: Do NOT include any mockups or product previews
            7. IMPORTANT: Create ONLY the logo graphic itself
            8. NO META REFERENCES - do not show the logo applied to anything
            9. Design should be a standalone graphic symbol/icon only
            10. CRITICAL: Clean vector art style with crisp lines and solid colors"""
            
            # 生成透明背景的矢量logo
            logo_image = generate_vector_image(logo_prompt)
        
        # 最终设计 - 不添加文字
        final_design = colored_shirt
        
        # 应用Logo (如果有)
        if logo_image:
            # 应用透明背景的logo到T恤
            final_design = apply_logo_to_shirt(colored_shirt, logo_image, "center", 60)
        
        return final_design, {
            "color": {"hex": color_hex, "name": color_name},
            "fabric": fabric_type,
            "logo": logo_description,
            "design_index": 0 if variation_id is None else variation_id  # 使用design_index替代variation_id
        }
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return None, {"error": f"Error generating design: {str(e)}\n{traceback_str}"}

def generate_single_design(design_index):
    try:
        # 为每个设计添加轻微的提示词变化，确保设计多样性
        design_variations = [
            "",  # 原始提示词
            "modern and minimalist",
            "colorful and vibrant",
            "vintage and retro",
            "elegant and simple"
        ]
        
        # 选择合适的变化描述词
        variation_desc = ""
        if design_index < len(design_variations):
            variation_desc = design_variations[design_index]
        
        # 创建变化的提示词
        if variation_desc:
            # 将变化描述词添加到原始提示词
            varied_prompt = f"{design_prompt}, {variation_desc}"
        else:
            varied_prompt = design_prompt
        
        # 完整的独立流程 - 每个设计独立获取AI建议、生成图片，确保颜色一致性
        # 使用独立提示词生成完全不同的设计
        design, info = generate_complete_design(varied_prompt)
        
        # 添加设计索引到信息中以便排序
        if info and isinstance(info, dict):
            info["design_index"] = design_index
        
        return design, info
    except Exception as e:
        print(f"Error generating design {design_index}: {e}")
        return None, {"error": f"Failed to generate design {design_index}"}

def generate_multiple_designs(design_prompt, count=1):
    """Generate multiple T-shirt designs in parallel - independent designs rather than variations"""
    if count <= 1:
        # 如果只需要一个设计，直接生成不需要并行
        base_design, base_info = generate_complete_design(design_prompt)
        if base_design:
            return [(base_design, base_info)]
        else:
            return []
    
    designs = []
    
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(count, 5)) as executor:
        # 提交所有任务
        future_to_id = {executor.submit(generate_single_design, i): i for i in range(count)}
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_id):
            design_id = future_to_id[future]
            try:
                design, info = future.result()
                if design:
                    designs.append((design, info))
            except Exception as e:
                print(f"Design {design_id} generated an exception: {e}")
    
    # 按照设计索引排序
    designs.sort(key=lambda x: x[1].get("design_index", 0) if x[1] and "design_index" in x[1] else 0)
    
    return designs

# ===== 模特试穿功能 =====

def save_image_temporarily(image, prefix="temp_image"):
    """将PIL图像保存为临时文件，返回文件路径"""
    try:
        # 创建临时文件夹
        temp_dir = "temp_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # 生成唯一的文件名
        unique_id = str(uuid.uuid4())[:8]
        temp_path = os.path.join(temp_dir, f"{prefix}_{unique_id}.png")
        
        # 保存图像
        image.save(temp_path, "PNG")
        
        return temp_path
    except Exception as e:
        print(f"Error saving temporary image: {e}")
        return None

def optimize_image_for_tryon(image):
    """
    优化图片使其更适合AI试衣
    
    Args:
        image: PIL图像对象
    
    Returns:
        优化后的PIL图像对象
    """
    try:
        print("开始优化图片格式...")
        
        # 1. 转换为RGB模式
        if image.mode == 'RGBA':
            # 创建白色背景
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为蒙版
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"图片模式已转换为: {image.mode}")
        
        # 2. 调整图片尺寸 - 确保符合API要求（150-4096像素）
        width, height = image.size
        
        # 如果图片太小，放大到合适尺寸
        if min(width, height) < 150:
            scale_factor = 150 / min(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            print(f"图片尺寸从 {width}x{height} 放大到 {new_width}x{new_height}")
        
        # 如果图片太大，缩小到合适尺寸
        elif max(width, height) > 2048:  # 使用2048而不是4096，确保处理效率
            scale_factor = 2048 / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            print(f"图片尺寸从 {width}x{height} 缩小到 {new_width}x{new_height}")
        
        # 3. 增强图片质量
        from PIL import ImageEnhance
        
        # 轻微增强对比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # 轻微增强清晰度
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.05)
        
        print(f"图片优化完成，最终尺寸: {image.size}")
        
        return image
        
    except Exception as e:
        print(f"图片优化失败: {e}")
        # 如果优化失败，返回原图
        return image

def upload_to_oss(image_path):
    """尝试将图片上传到阿里云 OSS，并返回公网 URL。

    需要以下环境变量（或在代码中硬编码，但强烈推荐使用环境变量）：
    - ALI_OSS_ENDPOINT  例如："oss-cn-hangzhou.aliyuncs.com"
    - ALI_OSS_BUCKET   例如："your-bucket-name"
    - ALI_OSS_AK       阿里云 AccessKeyId
    - ALI_OSS_SK       阿里云 AccessKeySecret
    """
    try:
        import oss2  # 确保在 requirements.txt 中添加 oss2
        import os as _os

        endpoint = _os.getenv("ALI_OSS_ENDPOINT")
        bucket_name = _os.getenv("ALI_OSS_BUCKET")
        access_key = _os.getenv("ALI_OSS_AK")
        secret_key = _os.getenv("ALI_OSS_SK")

        # 配置不完整就放弃 OSS
        if not (endpoint and bucket_name and access_key and secret_key):
            print("⚠️ OSS 环境变量未配置完整，跳过 OSS 上传")
            return None

        # 构建 OSS 客户端
        auth = oss2.Auth(access_key, secret_key)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        # object_name 使用 uuid 保证唯一
        object_name = f"tshirt-designs/{uuid.uuid4()}.png"

        print(f"开始上传至 OSS: {object_name}")
        bucket.put_object_from_file(object_name, image_path)

        # 生成公网 URL（注意 endpoint 可能带 https://）
        endpoint_netloc = endpoint.replace("https://", "").replace("http://", "")
        public_url = f"https://{bucket_name}.{endpoint_netloc}/{object_name}"
        print(f"✅ OSS 上传成功: {public_url}")
        return public_url

    except Exception as e:
        print(f"❌ OSS 上传失败: {e}")
        return None

def upload_image_to_get_public_url(image_path):
    """
    上传图像到获得公网URL
    使用多个可靠的免费图片托管服务
    """
    
    print(f"开始上传图片: {image_path}")
    
    # 尝试多个可靠的图片托管服务
    upload_services = [
        {
            "name": "imgbb",
            "url": "https://api.imgbb.com/1/upload",
            "method": "imgbb",
            "key": "2d1f44e048f7a69c02947e9ad0797e48"  # 公共API密钥
        },
        {
            "name": "catbox",
            "url": "https://catbox.moe/user/api.php",
            "method": "catbox"
        },
        {
            "name": "tmpfiles", 
            "url": "https://tmpfiles.org/api/v1/upload",
            "method": "tmpfiles"
        }
    ]
    
    for service in upload_services:
        try:
            print(f"尝试使用 {service['name']} 上传图片...")
            
            if service['method'] == 'imgbb':
                with open(image_path, 'rb') as file:
                    files = {'image': file}
                    data = {'key': service['key']}
                    
                    response = requests.post(service['url'], files=files, data=data, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            url = result['data']['url']
                            print(f"✅ {service['name']} 上传成功: {url}")
                            return url
                        else:
                            print(f"❌ {service['name']} 响应失败: {result}")
                            
            elif service['method'] == 'catbox':
                with open(image_path, 'rb') as file:
                    files = {'fileToUpload': file}
                    data = {'reqtype': 'fileupload'}
                    
                    response = requests.post(service['url'], files=files, data=data, timeout=30)
                    
                    if response.status_code == 200:
                        url = response.text.strip()
                        if url.startswith('http'):
                            print(f"✅ {service['name']} 上传成功: {url}")
                            return url
                        else:
                            print(f"❌ {service['name']} 响应异常: {url}")
                            
            elif service['method'] == 'tmpfiles':
                with open(image_path, 'rb') as file:
                    files = {'file': file}
                    
                    response = requests.post(service['url'], files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('status') == 'success':
                            # tmpfiles.org 返回的URL需要转换
                            url = result['data']['url'].replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                            print(f"✅ {service['name']} 上传成功: {url}")
                            return url
                        else:
                            print(f"❌ {service['name']} 响应失败: {result}")
                            
        except Exception as e:
            print(f"❌ {service['name']} 上传失败: {e}")
            continue
    
    # 所有上传服务都失败时的处理
    print("⚠️ all_upload_services_failed")
    st.warning("all_upload_services_failed")
    
    # 返回示例图片URL作为备选方案
    return "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250626/epousa/short_sleeve.jpeg"

def create_tryon_task(person_image_url, garment_image_url):
    """
    创建AI试衣任务 - 只试穿上装，模型随机生成下装
    
    Args:
        person_image_url: 模特图片的公网URL
        garment_image_url: 上装图片的公网URL
    
    Returns:
        dict: 包含task_id的响应或错误信息
    """
    try:
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis/"
        
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"
        }
        
        # 构建请求数据 - 按照API文档格式，只传入top_garment_url
        data = {
            "model": "aitryon-plus",
            "input": {
                "person_image_url": person_image_url,
                "top_garment_url": garment_image_url
            },
            "parameters": {
                "resolution": -1,
                "restore_face": True
            }
        }
        
        print(f"create_tryon_task: {data}")
        
        # 发送请求
        response = requests.post(url, headers=headers, json=data)
        
        print(f"create_tryon_task: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"create_tryon_task: {result}")
            return result
        else:
            error_text = response.text
            print(f"create_tryon_task: {error_text}")
            return {"error": f"API request failed with status {response.status_code}: {error_text}"}
            
    except Exception as e:
        print(f"create_tryon_task: {e}")
        return {"error": f"Error creating tryon task: {str(e)}"}

def poll_tryon_task(task_id, max_attempts=60, poll_interval=3, progress_callback=None):
    """
    轮询AI试衣任务状态
    
    Args:
        task_id: 任务ID
        max_attempts: 最大轮询次数
        poll_interval: 轮询间隔（秒）
        progress_callback: 进度回调函数，接受(attempt, max_attempts, status)参数
    
    Returns:
        dict: 任务结果或错误信息
    """
    try:
        url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
        }
        
        for attempt in range(max_attempts):
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                task_status = result.get("output", {}).get("task_status", "UNKNOWN")
                
                print(f"轮询第 {attempt + 1} 次，任务状态: {task_status}")
                
                # 调用进度回调
                if progress_callback:
                    progress_callback(attempt + 1, max_attempts, task_status)
                
                if task_status == "SUCCEEDED":
                    print("poll_tryon_task: SUCCEEDED")
                    return result
                elif task_status == "FAILED":
                    error_msg = result.get('output', {}).get('message', 'Unknown error')
                    print(f"poll_tryon_task: {error_msg}")
                    return {"error": f"Task failed: {error_msg}"}
                elif task_status in ["CANCELED", "UNKNOWN"]:
                    print(f"poll_tryon_task: {task_status}")
                    return {"error": f"Task {task_status.lower()}"}
                else:
                    # 任务仍在进行中，继续轮询
                    print(f"poll_tryon_task: {task_status}")
                    time.sleep(poll_interval)
            else:
                print(f"poll_tryon_task: {response.status_code}, {response.text}")
                return {"error": f"Polling failed with status {response.status_code}: {response.text}"}
        
        print(f"poll_tryon_task: {max_attempts}")
        return {"error": "Task polling timeout"}
        
    except Exception as e:
        return {"error": f"Error polling task: {str(e)}"}

def generate_model_tryon(tshirt_image, model_image_url=None, progress_callback=None):
    """
    生成模特试穿效果
    
    Args:
        tshirt_image: PIL图像对象，T恤设计图片
        model_image_url: 模特图片的公网URL，如果为None则使用默认模特
        progress_callback: 进度回调函数，接受(progress, message)参数
    
    Returns:
        tuple: (试穿效果图PIL对象, 状态信息dict)
    """
    def update_progress(progress, message):
        if progress_callback:
            progress_callback(progress, message)
        print(f"进度 {progress}%: {message}")
    try:
        # 如果没有提供模特图片，使用默认模特
        if model_image_url is None:
            # 使用阿里云文档中提供的示例模特图
            model_image_url = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250626/ubznva/model_person.png"
        
        update_progress(10, "generate_model_tryon: start")
        print(f"generate_model_tryon: {model_image_url}")
        
        # 1. 保存T恤图片到临时文件
        update_progress(15, "generate_model_tryon: save_image_temporarily")
        temp_path = save_image_temporarily(tshirt_image, "tshirt_design")
        if not temp_path:
            return None, {"error": "Failed to save temporary image"}
        
        print(f"generate_model_tryon: {temp_path}")
        
        # 2. 优化T恤图片并上传获得公网URL
        update_progress(20, "generate_model_tryon: optimize_image_for_tryon")
        # 先将图片转换为适合试衣的格式
        optimized_image = optimize_image_for_tryon(tshirt_image)
        
        # 保存优化后的图片
        optimized_path = save_image_temporarily(optimized_image, "optimized_tshirt")
        if not optimized_path:
            print("save_image_temporarily: Failed to save optimized image")
            return None, {"error": "Failed to save optimized image"}
        
        print(f"generate_model_tryon: {optimized_path}")
        
        update_progress(30, "upload_image_to_get_public_url")
        # 上传优化后的图片
        garment_url = upload_image_to_get_public_url(optimized_path)
        if not garment_url or garment_url == "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250626/epousa/short_sleeve.jpeg":
            # 如果上传失败，尝试上传原始图片
            print(" Failed to upload optimized image, trying to upload original image")
            garment_url = upload_image_to_get_public_url(temp_path)
        
        print(f"generate_model_tryon: {garment_url}")
        
        # 显示上传状态
        if garment_url != "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250626/epousa/short_sleeve.jpeg":
            update_progress(40, "✅ your t-shirt design has been successfully uploaded, using your actual design!")
        else:
            update_progress(40, "⚠️ image upload failed, using example image for demonstration")
        
        # 清理优化图片的临时文件
        try:
            if optimized_path != temp_path:
                os.remove(optimized_path)
        except:
            pass
        
        # 3. 创建试衣任务（只试穿上装）
        update_progress(45, "create_tryon_task")
        print("create_tryon_task")
        task_response = create_tryon_task(model_image_url, garment_url)
        
        if "error" in task_response:
            print(f"create_tryon_task: {task_response}")
            return None, task_response
        
        task_id = task_response.get("output", {}).get("task_id")
        if not task_id:
            print(f"无法获取任务ID: {task_response}")
            return None, {"error": "Failed to get task ID from response"}
        
        print(f"create_tryon_task: task created successfully, task ID: {task_id}")
        update_progress(50, "AI is processing the try-on effect...")
        
        # 4. 轮询任务状态
        print("poll_tryon_task")
        
        # 创建轮询进度回调
        def poll_progress_callback(attempt, max_attempts, status):
            progress = 50 + int(40 * attempt / max_attempts)  # 50-90%的进度用于轮询
            update_progress(progress, f"生成中...（状态: {status}）")
        
        result = poll_tryon_task(task_id, progress_callback=poll_progress_callback)
        
        if "error" in result:
            print(f"任务执行失败: {result}")
            return None, result
        
        # 5. 下载试穿效果图
        update_progress(90, "download_tryon_image")
        image_url = result.get("output", {}).get("image_url")
        if not image_url:
            print(f"generate_model_tryon: {result}")
            return None, {"error": "No image URL in result"}
        
        print(f"generate_model_tryon: {image_url}")
        
        # 下载图片
        img_response = requests.get(image_url, timeout=30)
        if img_response.status_code == 200:
            try_on_image = Image.open(BytesIO(img_response.content)).convert("RGBA")
            
            print(f"generate_model_tryon: {try_on_image.size}")
            update_progress(100, "✅ try-on effect generated successfully")
            
            # 清理临时文件
            try:
                os.remove(temp_path)
                print(f"临时文件已清理: {temp_path}")
            except:
                pass
            
            return try_on_image, {
                "success": True,
                "task_id": task_id,
                "image_url": image_url,
                "message": "try-on effect generated successfully"
            }
        else:
            print(f"下载图片失败，状态码: {img_response.status_code}")
            return None, {"error": f"Failed to download result image: {img_response.status_code}"}
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"生成试穿效果时发生错误: {error_details}")
        return None, {"error": f"Error in model tryon: {str(e)}\n{error_details}"}

# ===== 模特试穿功能结束 =====

def show_high_recommendation_without_explanation():
    st.title("👕 AI Recommendation Experiment Platform")
    st.markdown("### Study1-Let AI Design Your T-shirt")
    
    # 显示实验组和设计数量信息
    st.info(f"You are currently in Study1, and AI will generate {DEFAULT_DESIGN_COUNT} T-shirt design options for you")
    
    # 初始化会话状态变量
    if 'user_prompt' not in st.session_state:
        st.session_state.user_prompt = ""
    if 'final_design' not in st.session_state:
        st.session_state.final_design = None
    if 'design_info' not in st.session_state:
        st.session_state.design_info = None
    if 'is_generating' not in st.session_state:
        st.session_state.is_generating = False
    if 'should_generate' not in st.session_state:
        st.session_state.should_generate = False
    if 'recommendation_level' not in st.session_state:
        # 设置固定推荐级别，不再允许用户选择
        if DEFAULT_DESIGN_COUNT == 1:
            st.session_state.recommendation_level = "low"
        elif DEFAULT_DESIGN_COUNT == 3:
            st.session_state.recommendation_level = "medium"
        else:  # 5或其他值
            st.session_state.recommendation_level = "high"
    if 'generated_designs' not in st.session_state:
        st.session_state.generated_designs = []
    if 'selected_design_index' not in st.session_state:
        st.session_state.selected_design_index = 0
    if 'tryon_result' not in st.session_state:
        st.session_state.tryon_result = None
    if 'tryon_info' not in st.session_state:
        st.session_state.tryon_info = None
    if 'is_generating_tryon' not in st.session_state:
        st.session_state.is_generating_tryon = False
    if 'original_tshirt' not in st.session_state:
        # 加载原始白色T恤图像
        try:
            original_image_path = "white_shirt.png"
            possible_paths = [
                "white_shirt.png",
                "./white_shirt.png",
                "../white_shirt.png",
                "images/white_shirt.png",
            ]
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    original_image_path = path
                    found = True
                    break
            
            if found:
                st.session_state.original_tshirt = Image.open(original_image_path).convert("RGBA")
            else:
                st.error("Could not find base T-shirt image")
                st.session_state.original_tshirt = None
        except Exception as e:
            st.error(f"Error loading T-shirt image: {str(e)}")
            st.session_state.original_tshirt = None
    
    # 创建三列布局：设计区、试穿效果区、输入区
    design_col, tryon_col, input_col = st.columns([2, 2, 2])
    
    with design_col:
        # 创建占位区域用于T恤设计展示
        design_area = st.empty()
        
        # 在设计区域显示当前状态的T恤设计
        if st.session_state.final_design is not None:
            with design_area.container():
                st.markdown("### Your Custom T-shirt Design")
                st.image(st.session_state.final_design, use_container_width=True)
        elif len(st.session_state.generated_designs) > 0:
            with design_area.container():
                st.markdown("### Generated Design Options")
                
                # 创建多列来显示设计
                design_count = len(st.session_state.generated_designs)
                if design_count > 3:
                    # 两行显示
                    row1_cols = st.columns(min(3, design_count))
                    row2_cols = st.columns(min(3, max(0, design_count - 3)))
                    
                    # 显示第一行
                    for i in range(min(3, design_count)):
                        with row1_cols[i]:
                            design, _ = st.session_state.generated_designs[i]
                            st.markdown(f"<p style='text-align:center;'>Design {i+1}</p>", unsafe_allow_html=True)
                            # 显示设计
                            st.image(design, use_container_width=True)
                    
                    # 显示第二行
                    for i in range(3, design_count):
                        with row2_cols[i-3]:
                            design, _ = st.session_state.generated_designs[i]
                            st.markdown(f"<p style='text-align:center;'>Design {i+1}</p>", unsafe_allow_html=True)
                            # 显示设计
                            st.image(design, use_container_width=True)
                else:
                    # 单行显示
                    cols = st.columns(design_count)
                    for i in range(design_count):
                        with cols[i]:
                            design, _ = st.session_state.generated_designs[i]
                            st.markdown(f"<p style='text-align:center;'>Design {i+1}</p>", unsafe_allow_html=True)
                            # 显示设计
                            st.image(design, use_container_width=True)
                

        else:
            # 显示原始空白T恤
            with design_area.container():
                st.markdown("### T-shirt Design Preview")
                if st.session_state.original_tshirt is not None:
                    st.image(st.session_state.original_tshirt, use_container_width=True)
                else:
                    st.info("Could not load original T-shirt image, please refresh the page")
    
    with tryon_col:
        # 模特试穿效果展示区
        st.markdown("### Model Try-on Effect")
        
        if st.session_state.tryon_result is not None:
            st.image(st.session_state.tryon_result, use_container_width=True)
            if st.session_state.tryon_info and "message" in st.session_state.tryon_info:
                st.success(st.session_state.tryon_info["message"])
        elif st.session_state.is_generating_tryon:
            st.info("🤖 AI is generating the try-on effect, please wait...")
            st.image("https://via.placeholder.com/400x600/f0f0f0/999999?text=generating...", use_container_width=True)
        else:
            st.info("👕 Please generate a t-shirt design first, then click 'Generate Model Try-on' to view the wearing effect")
            st.image("https://via.placeholder.com/400x600/f0f0f0/999999?text=试穿预览", use_container_width=True)
    
    with input_col:
        # 设计提示词和推荐级别选择区
        st.markdown("### Design Options")
        
        # # 移除推荐级别选择按钮，改为显示当前级别信息
        # if DEFAULT_DESIGN_COUNT == 1:
        #     level_text = "Low - will generate 1 design"
        # elif DEFAULT_DESIGN_COUNT == 3:
        #     level_text = "Medium - will generate 3 designs"
        # else:  # 5或其他值
        #     level_text = "High - will generate 5 designs"
            
        # st.markdown(f"""
        # <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 20px;">
        # <p style="margin: 0; font-size: 16px; font-weight: bold;">Current recommendation level: {level_text}</p>
        # </div>
        # """, unsafe_allow_html=True)
        
        # 提示词输入区
        st.markdown("#### Describe your desired T-shirt design:")
        
        # 添加简短说明
        st.markdown("""
        <div style="margin-bottom: 15px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
        <p style="margin: 0; font-size: 14px;">Enter three keywords to describe your ideal T-shirt design. 
        Our AI will combine these features to create unique designs for you.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 初始化关键词状态
        if 'keywords' not in st.session_state:
            st.session_state.keywords = ""
        
        # 关键词输入框
        keywords = st.text_input("Enter keywords for your design", value=st.session_state.keywords, 
                              placeholder="e.g., casual, nature, blue", key="input_keywords")
        
        # 生成设计按钮
        generate_col = st.empty()
        with generate_col:
            generate_button = st.button("🎨 Generate T-shirt Design", key="generate_design", use_container_width=True)
        
        # 模特试穿按钮
        st.markdown("---")
        st.markdown("#### Model Try-on")
        st.markdown("""
        <div style="margin-bottom: 15px; padding: 10px; background-color: #e8f4fd; border-radius: 5px; border-left: 4px solid #0066cc;">
        <p style="margin: 0; font-size: 14px;">🎭 Use AI try-on technology to make your t-shirt design look like it's on a real person!<br/>
        ✨ Your actual design image will be uploaded to the cloud for try-on<br/>
        ⏱️ The generation process takes 15-30 seconds, please wait patiently.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 检查是否有可用的设计
        can_generate_tryon = (st.session_state.final_design is not None or 
                             len(st.session_state.generated_designs) > 0)
        
        if can_generate_tryon:
            # 如果有多个设计，让用户选择要试穿的设计
            if len(st.session_state.generated_designs) > 1:
                selected_design_index = st.selectbox(
                    "Choose design for try-on:", 
                    range(len(st.session_state.generated_designs)),
                    format_func=lambda x: f"Design {x+1}",
                    key="tryon_design_select"
                )
            else:
                selected_design_index = 0
                
            tryon_button = st.button("👗 Generate Model Try-on", 
                                   key="generate_tryon", 
                                   use_container_width=True,
                                   disabled=st.session_state.is_generating_tryon)
        else:
            st.info("Please generate a t-shirt design first, then click 'Generate Model Try-on' to view the wearing effect")
            tryon_button = False
        
        # 创建进度和消息区域在输入框下方
        progress_area = st.empty()
        message_area = st.empty()
        tryon_progress_area = st.empty()
        tryon_message_area = st.empty()
        
        # 生成设计按钮事件处理
        if generate_button:
            # 保存用户输入的关键词
            st.session_state.keywords = keywords
            
            # 检查是否输入了关键词
            if not keywords:
                st.error("Please enter at least one keyword")
            else:
                # 直接使用用户输入的关键词作为提示词
                user_prompt = keywords
                
                # 保存用户输入
                st.session_state.user_prompt = user_prompt
                
                # 使用固定的设计数量
                design_count = DEFAULT_DESIGN_COUNT
                
                # 清空之前的设计
                st.session_state.final_design = None
                st.session_state.generated_designs = []
                
                try:
                    # 显示生成进度
                    with design_area.container():
                        st.markdown("### Generating T-shirt Designs")
                        if st.session_state.original_tshirt is not None:
                            st.image(st.session_state.original_tshirt, use_container_width=True)
                    
                    # 创建进度条和状态消息在输入框下方
                    progress_bar = progress_area.progress(0)
                    message_area.info(f"AI is generating {design_count} unique designs for you. This may take about a minute. Please do not refresh the page or close the browser. Thank you for your patience! ♪(･ω･)ﾉ")
                    # 记录开始时间
                    start_time = time.time()
                    
                    # 收集生成的设计
                    designs = []
                    
                    # 生成单个设计的安全函数
                    def generate_single_safely(design_index):
                        try:
                            return generate_complete_design(user_prompt, design_index)
                        except Exception as e:
                            message_area.error(f"Error generating design: {str(e)}")
                            return None, {"error": f"Failed to generate design: {str(e)}"}
                    
                    # 对于单个设计，直接生成
                    if design_count == 1:
                        design, info = generate_single_safely(0)
                        if design:
                            designs.append((design, info))
                        progress_bar.progress(100)
                        message_area.success("Design generation complete!")
                    else:
                        # 为多个设计使用并行处理
                        completed_count = 0
                        
                        # 进度更新函数
                        def update_progress():
                            nonlocal completed_count
                            completed_count += 1
                            progress = int(100 * completed_count / design_count)
                            progress_bar.progress(progress)
                            message_area.info(f"Generated {completed_count}/{design_count} designs...")
                        
                        # 使用线程池并行生成多个设计
                        with concurrent.futures.ThreadPoolExecutor(max_workers=design_count) as executor:
                            # 提交所有任务
                            future_to_id = {executor.submit(generate_single_safely, i): i for i in range(design_count)}
                            
                            # 收集结果
                            for future in concurrent.futures.as_completed(future_to_id):
                                design_id = future_to_id[future]
                                try:
                                    design, info = future.result()
                                    if design:
                                        designs.append((design, info))
                                except Exception as e:
                                    message_area.error(f"Design {design_id} generation failed: {str(e)}")
                                
                                # 更新进度
                                update_progress()
                        
                        # 按照ID排序设计
                        designs.sort(key=lambda x: x[1].get("design_index", 0) if x[1] and "design_index" in x[1] else 0)
                    
                    # 记录结束时间
                    end_time = time.time()
                    generation_time = end_time - start_time
                    
                    # 存储生成的设计
                    if designs:
                        st.session_state.generated_designs = designs
                        st.session_state.selected_design_index = 0
                        message_area.success(f"Generated {len(designs)} designs in {generation_time:.1f} seconds!")
                    else:
                        message_area.error("Could not generate any designs. Please try again.")
                    
                    # 重新渲染设计区域以显示新生成的设计
                    st.rerun()
                except Exception as e:
                    import traceback
                    message_area.error(f"An error occurred: {str(e)}")
                    st.error(traceback.format_exc())
    
        # 模特试穿按钮事件处理
        if tryon_button:
            # 确定要试穿的设计
            design_to_tryon = None
            
            if st.session_state.final_design is not None:
                design_to_tryon = st.session_state.final_design
            elif len(st.session_state.generated_designs) > 0:
                if 'selected_design_index' in locals():
                    design_to_tryon = st.session_state.generated_designs[selected_design_index][0]
                else:
                    design_to_tryon = st.session_state.generated_designs[0][0]
            
            if design_to_tryon is not None:
                # 清空之前的试穿结果
                st.session_state.tryon_result = None
                st.session_state.tryon_info = None
                st.session_state.is_generating_tryon = True
                
                # 显示进度
                tryon_progress_bar = tryon_progress_area.progress(0)
                tryon_message_area.info("🤖 AI is generating the try-on effect...")
                
                # 显示详细状态
                status_placeholder = st.empty()
                status_placeholder.warning("📋 Preparing stage: processing your t-shirt design image...")
                
                try:
                    # 更新进度到10%
                    tryon_progress_bar.progress(10)
                    status_placeholder.info("🖼️ Optimizing image format and size...")
                    
                    # 更新进度到25%
                    tryon_progress_bar.progress(25)
                    status_placeholder.info("☁️ Uploading your t-shirt design to the cloud...")
                    
                    # 更新进度到40%
                    tryon_progress_bar.progress(40)
                    status_placeholder.info("🚀 Creating AI try-on task...")
                    
                    # 创建进度回调函数
                    def progress_update(progress, message):
                        tryon_progress_bar.progress(min(progress, 95))  # 限制最大进度到95%
                        status_placeholder.info(f"{message}")
                    
                    # 调用模特试穿功能
                    tryon_result, tryon_info = generate_model_tryon(design_to_tryon, progress_callback=progress_update)
                    
                    # 更新进度到100%
                    tryon_progress_bar.progress(100)
                    
                    # 更新状态
                    st.session_state.tryon_result = tryon_result
                    st.session_state.tryon_info = tryon_info
                    st.session_state.is_generating_tryon = False
                    
                    if tryon_result is not None:
                        tryon_message_area.success("✅ Model try-on effect generated successfully!")
                        status_placeholder.success("🎉 Try-on effect generated, please check the preview area on the right")
                        tryon_progress_area.empty()
                    else:
                        error_msg = tryon_info.get("error", "Unknown error") if tryon_info else "Unknown error"
                        tryon_message_area.error(f"❌ Failed to generate try-on effect: {error_msg}")
                        status_placeholder.error("💥 Failed to generate try-on effect, please check the network connection or try again later")
                        tryon_progress_area.empty()
                    
                    # 重新渲染页面
                    time.sleep(1)  # 让用户看到完成状态
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.is_generating_tryon = False
                    import traceback
                    error_details = traceback.format_exc()
                    tryon_message_area.error(f"❌ Try-on generation error: {str(e)}")
                    print(f"Try-on generation error: {error_details}")
                    tryon_progress_area.empty()
            else:
                tryon_message_area.error("No available design for try-on, please generate a t-shirt design first.")
    

