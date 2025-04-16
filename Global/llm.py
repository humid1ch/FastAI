"""deepseek_by_ds官方的api"""
from dashscope import api_key


def deepseek_by_ds(model:str='deepseek-reasoner' ):
    """官方的api"""
    import io 
    import json
    from langchain_openai.chat_models.base import BaseChatOpenAI
    Path="./config/key"
    with io.open(Path,'r') as file:
        josn_data_text=file.read()
    json_data=json.loads(josn_data_text)
    #deepseek 官方的api
    return BaseChatOpenAI(
        model=model,  # 使用DeepSeek聊天模型
        api_key=json_data["deepseek_key"],  # 替换为你的API易API密钥
        openai_api_base='https://api.deepseek.com',  # API易的端点
        max_tokens=1024  # 设置最大生成token数
        )
"""deepseek_by_ds官方的api"""


"""deepseek_r_by_ollama本地的ollama"""
def deepseek_r_by_ollama(url:str="http://localhost:11434"):
    from langchain_ollama import OllamaLLM 
    # 初始化Ollama模型（以llama2为例）
    return OllamaLLM(
        model="deepseek-r1:32b",  # 模型名称，需与Ollama本地模型一致 
        temperature=0.8,  # 控制生成多样性 
        base_url=url # Ollama服务地址 
    )
"""deepseek_by_ollama本地的ollama"""


"""qwen_vl_max 百炼的api"""
def mutil_llm(model:str = "qwen-vl-max-latest"):
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

    #初始化
    Path="./config/key"
    with io.open(Path,'r') as file:
        json_data_text=file.read()
    json_data=json.loads(json_data_text)

    """qwen_vl_max 百炼的api"""
    # return ChatOpenAI(
    #     model=model,
    #     api_key=json_data["qwen_vl_key"],
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    # )
    # return ChatOpenAI(
    #     api_key="03493064-4329-4bd3-99ba-85e5cd2299e0",
    #     base_url="https://ark.cn-beijing.volces.com/api/v3",
    #     model="doubao-vision-pro-32k-241028"
    # )
    return ChatOpenAI(
        api_key="sk-1a9b0010300a4ff6b7b909a00754ea7c",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # model="qvq-max",
        # streaming=True
        # model="qwen-vl-plus",
        model="qwen-vl-max"
        # model="qwen-vl-ocr"
        # model="qwen2.5-vl-32b-instruct"
        # model="qwen2-vl-72b-instruct"
        # model="llama-4-maverick-17b-128e-instruct"
    )
"""qwen_vl_max 百炼的api"""