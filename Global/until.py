def baseimage(data):
    #默认编码器
    import base64
    return base64.b64encode(data).decode("utf-8")

def adjust_image(image_path, model_name="qwen-vl-max" , encodeFun = baseimage, output_path=None,max_attempts=3):
    """
    args
        image_path 图片路径
        model_name 模型名字
        output_path 保存路径 默认为空，为空则不保存
        encodeFun   编码的函数
        max_attempts 压缩次数
    return bytes
    自动调整图片的宽高比像素
    只适配了千问系列
            model_name 为一下值
                "qwen-vl-max", "qwen-vl-max-latest", "qwen-vl-max-1230", 
                "qwen-vl-max-1119", "qwen-vl-max-1030", "qwen-vl-max-0809",
                "qwen-vl-plus-latest", "qwen-vl-plus-0102", "qwen-vl-plus-0809",
                "qwen2-vl-72b-instruct", "qwen2-vl-7b-instruct", "qwen2-vl-2b-instruct"
                "qwen-vl-max-0201", "qwen-vl-plus"
    """
    import io 
    from PIL import Image 
    # 定义模型像素要求 
    MODEL_REQUIREMENTS = {
        "12m_models": {
            "names": [
                "qwen-vl-max", "qwen-vl-max-latest", "qwen-vl-max-1230", 
                "qwen-vl-max-1119", "qwen-vl-max-1030", "qwen-vl-max-0809",
                "qwen-vl-plus-latest", "qwen-vl-plus-0102", "qwen-vl-plus-0809",
                "qwen2-vl-72b-instruct", "qwen2-vl-7b-instruct", "qwen2-vl-2b-instruct"
            ],
            "max_pixels": 12000000 
        },
        "1m_models": {
            "names": ["qwen-vl-max-0201", "qwen-vl-plus"],
            "max_pixels": 1048576 
        }
    }
 
    try:
        # 1. 检查并处理文件大小 
        img = Image.open(image_path) 
        original_format = img.format  
        
        # 转换为RGB模式(如果是RGBA或P模式)
        if img.mode  in ('RGBA', 'P'):
            img = img.convert('RGB') 
        
        # 2. 调整尺寸(根据模型要求)
        width, height = img.size  
        
        # 检查最小尺寸 
        if width <= 10 or height <= 10:
            raise ValueError(f"图片尺寸过小({width}x{height})，至少需要10x10像素")
        
        # 检查宽高比 
        aspect_ratio = width / height 
        if aspect_ratio > 200 or aspect_ratio < 1/200:
            # 自动裁剪到最大允许宽高比 
            if aspect_ratio > 200:
                new_width = int(height * 200)
                left = (width - new_width) // 2 
                img = img.crop((left,  0, left + new_width, height))
            else:
                new_height = int(width * 200)
                top = (height - new_height) // 2 
                img = img.crop((0,  top, width, top + new_height))
        
        # 根据模型调整尺寸 
        max_pixels = None 
        for req in MODEL_REQUIREMENTS.values(): 
            if model_name in req["names"]:
                max_pixels = req["max_pixels"]
                break 
        
        if max_pixels:
            current_pixels = img.width  * img.height  
            if current_pixels > max_pixels:
                # 计算调整比例 
                ratio = (max_pixels / current_pixels) ** 0.5 
                new_size = (int(img.width  * ratio), int(img.height  * ratio))
                img = img.resize(new_size,  Image.LANCZOS)
        
        # 3. 压缩图片到10MB以下 
        output_buffer = io.BytesIO()
        quality = 95  # 初始质量 
        
        for attempt in range(max_attempts):
            output_buffer.seek(0) 
            output_buffer.truncate() 
            
            # 保存为JPEG(通常比PNG小)
            img.save(output_buffer,  format='JPEG', quality=quality, optimize=True)
            
            if output_buffer.tell()  <= 10 * 1024 * 1024:
                break 
                
            # 减小质量 
            quality = max(10, quality - 15)
        else:
            raise ValueError("无法在尝试次数内将图片压缩到10MB以下")
        
        # 4. 保存或返回结果 
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(output_buffer.getvalue()) 
            print(f"图片已保存到: {output_path}")
        
        if not encodeFun:
            return output_buffer.getvalue()
        else:
            return  encodeFun(output_buffer.getvalue())
    
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return None 
 
def extract_json_raw(text): 
    """
    从文本中提取原始JSON字符串（不解析）
    
    参数:
        text (str): 可能包含JSON的文本 
        
    返回:
        str: 提取到的原始JSON字符串，若无则返回None 
    """
    import regex 
    json_pattern = r'(\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\])'
    match = regex.search(json_pattern,  text, regex.DOTALL)
    return match.group(0)  if match else None 

def embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings 
    return  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",cache_folder="D:/wr/langchain/src/models",model_kwargs = {'device': 'cuda'})

def extract_paragraphs(content_str, json_str):
    """
        根据之前约定的json格式划分原始文本
    """
    import json
    # 解析JSON字符串
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string")
    
    # 获取data列表
    data_list = data.get("data", [])
    if not data_list:
        return []
    
    result = []
    
    for item in data_list:
        b = item.get("b", "")
        e = item.get("e", "")
        
        if not b or not e:
            continue
        
        # 查找开始和结束位置
        start_idx = content_str.find(b)
        end_idx = content_str.find(e)
        
        if start_idx == -1 or end_idx == -1:
            continue
        
        # 计算结束位置，加上结束字符串的长度
        end_idx += len(e)
        
        # 提取段落
        paragraph = content_str[start_idx:end_idx]
        result.append(paragraph)
    
    return result