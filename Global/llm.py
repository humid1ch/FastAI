# deepseek 官方api
def deepseek_by_ds(model: str = "deepseek-reasoner"):
    """官方的api"""
    import io
    import json
    from langchain_openai.chat_models.base import BaseChatOpenAI

    Path = "./config/key"
    with io.open(Path, "r") as file:
        json_data_text = file.read()
    json_data = json.loads(json_data_text)

    # deepseek 官方的api
    return BaseChatOpenAI(
        model=model,  # 使用DeepSeek聊天模型
        api_key=json_data["deepseek_key"],  # 替换为你的API易API密钥
        base_url="https://api.deepseek.com",  # API易的端点
        # max_tokens=1024  # 设置最大生成token数
    )


# ollama本地模型, 默认 deepseek-r1:32b qwen蒸馏
def deepseek_r_by_ollama(
    url: str = "http://localhost:11434", model: str = "deepseek-r1:32b"
):
    """
    args:
        url: ollama服务地址
        model: ollama本地模型名
    """
    from langchain_ollama import OllamaLLM

    # 初始化Ollama模型
    return OllamaLLM(
        model=model,  # 模型名称，需与Ollama本地模型一致
        temperature=0.8,  # 控制生成多样性
        base_url=url,  # Ollama服务地址
    )


# 多模态大模型选择
def mutil_llm(model: str = "qwen-vl-max-latest"):
    """
    qwen_vl_max多模态模型
    content=[
              {"type": "image_url"
              , "image_url": {"url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg" }},
              {"type": "text", "text": "图中描绘的是什么景象?"}
            ]
    """
    from langchain_openai import ChatOpenAI
    import io
    import json

    # 阿里云百炼视觉模型
    ALI_BAILIAN_MUTIL_MODELS = [
        "qwen-vl-max",
        "qwen-vl-max-latest",
        "qwen-vl-max-2025-01-25",
        "qwen-vl-max-2025-04-02",
        "qwen-vl-max-1030",
        "qwen-vl-max-1119",
        "qwen-vl-max-1230",
        "qwen-vl-max-0809",
        "qwen-vl-plus",
        "qwen-vl-plus-latest",
        "qwen-vl-plus-2025-01-25",
        "qwen-vl-plus-0102",
        "qwen-vl-plus-0809",
        "qwen2.5-vl-32b-instruct",
        "qwen2.5-vl-72b-instruct",
        "qwen2.5-vl-7b-instruct",
        "qwen2.5-vl-3b-instruct",
        "qwen2-vl-72b-instruct",
        "qwen2-vl-7b-instruct",
        "qwen2-vl-2b-instruct",
        "qwen-vl-ocr",
        "qwen-vl-ocr-latest",
        "qwen-vl-ocr-1028",
        "llama-4-scout-17b-16e-instruct",
        "llama-4-maverick-17b-128e-instruct",
    ]

    # 豆包视觉模型, 需要账号开通才能使用
    DOUBAO_MUTIL_MODELS = [
        "doubao-1.5-vision-pro-32k-250115",
        "doubao-vision-pro-32k-241028",
        "doubao-vision-lite-32k-241015",
    ]

    # 初始化
    Path = "./config/key"
    with io.open(Path, "r") as file:
        json_data_text = file.read()
    json_data = json.loads(json_data_text)

    if model in ALI_BAILIAN_MUTIL_MODELS:
        return ChatOpenAI(
            api_key=json_data["ali_bailian_key"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=model,
        )

    if model in DOUBAO_MUTIL_MODELS:
        return ChatOpenAI(
            api_key=json_data["doubao_key"],
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model=model,
        )
