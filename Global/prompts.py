from langchain_core.prompts import ChatPromptTemplate
def AbstractsPrompt(topic:str,text:str,format:str):
        """
        摘要提示词
        
        args
                topic 标题
                text 文本
                format 格式
        Prompt
                system:  摘要用户这个关于{topic}的文档,格式如下{format}
                user:    帮我总结这个文档的内容，文档内容：{text}
        """
        SystemMessagePromptT="摘要用户这个关于{topic}的文档,格式如下{format}"
        HumanMessagePromptT="帮我总结这个文档的内容，文档内容：{text}。"
        ChatPromptT= ChatPromptTemplate.from_messages([("system", SystemMessagePromptT),("user",HumanMessagePromptT)])
        return ChatPromptT.invoke({"topic":topic,"text":text,"format":format})
 
def ImagePrompt_by_url(image_url:str,query:str):
        """
        传输图片提示词

        args
                image_url 路径
                quary user的提示词

        Prompt
                system:  You are a helpful assistant.
                user:   [
                        {"type": "image_url", "image_url": {"url": f"{image_url}"}},
                        {"type": "text", "text": "{query}"}
                        ]
        """
        ChatPromptT = ChatPromptTemplate.from_messages([ 
        ("system", "You are a helpful assistant."),
        ("user", [
                {"type": "image_url", "image_url": {"url": f"{image_url}"}},
                {"type": "text", "text": "{query}"}
        ])
        ])
        return ChatPromptT.invoke({"query": query})  

def ImagePrompt_by_path(path: str,query: str ,model:str= "qwen-vl-max"):
        """
        自动编码传输图片文件

        args
                path 路径
                quary user的提示词
                model 调整为这个模型支持的格式
        model_name
                "qwen-vl-max", "qwen-vl-max-latest", "qwen-vl-max-1230",
                "qwen-vl-max-1119", "qwen-vl-max-1030", "qwen-vl-max-0809",
                "qwen-vl-plus-latest", "qwen-vl-plus-0102", "qwen-vl-plus-0809",
                "qwen2-vl-72b-instruct", "qwen2-vl-7b-instruct", "qwen2-vl-2b-instruct"
                "qwen-vl-max-0201", "qwen-vl-plus"
        Prompt
                system:  You are a helpful assistant.
                user:   [
                        {"type": "image_url", "image_url": {"url": f"{image_url}"}},
                        {"type": "text", "text": "{query}"}
                        ]
        """
        import Global.until.until as imagedata
        image_data=imagedata.adjust_image(path,model)
        ChatPromptT = ChatPromptTemplate.from_messages([ 
                ("system", "You are a helpful assistant."),
                ("user", [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", "text": "{query}"}
                ])
        ])
        return ChatPromptT.invoke({"query": query})  

def txtsliptPrompt1(text:str,length:int,key:str,format:str="""{
  "length": 文本的长度,
  "data": [
    {
      "s": "压缩语义1",
      "b": 划分起点1,
      "e": 划分终点1
    },
    {
      "s": "压缩语义2",
      "b": 划分起点2,
      "e": 划分终点2
    }
  ]
}"""):
        """
        ai按ai选定的固定大小划分
        args
                text 被分割的文本
                key  建议ai保存的关键字
                format json格式        被修改就要重新测试
        Prompt
        """
        ChatPromptT = ChatPromptTemplate.from_messages([ 
                ("system", "你是一个按照语义划分文本并压缩这些按照语义划分的文本内容的函数，你要按照给出的json格式返回对应的字符内容"),
                ("user",
                 """
                 按一下步骤执行文本任务
                        1按照标点符号、文本语义划分长度为{length}的文本,划分大小你自己确定,每个划分块彼此保留一些重叠的部分。
                        2提出每个文本的:压缩语义与原始文本 （压缩语义保留关键字：{key}）。
                        3按照指定的json格式输出:{format}         
                txt:{text}
                 """ 
                 )
                ])
        return ChatPromptT.invoke({"text": text,"length": length,"key":key,"format":format})  

def txtsliptPrompt2(text: str, length: int, key: str , Example :str= """4. 返回格式必须严格遵循：
        {
            "length": 总字符数,
            "data": [
                {
                    "s": "摘要",
                    "b": 起始位置,
                    "e": 结束位置
                },
                ...
            ]
        }

    示例输入：
    "深度学习需要大量数据。但小样本学习正在兴起。迁移学习可以减少数据依赖。"
    
    示例输出：
    {
        "length": 38,
        "data": [
            {
                "s": "深度学习的缺点",
                "b": 0,
                "e": 11
            },
            {
                "s": "小样本学习趋势",
                "b": 11,
                "e": 21
            },
            {
                "s": "迁移学习的优势",
                "b": 21,
                "e": 38
            }
        ]
    }
                    """):
    """
    args:
        text: 被分割的文本
        length: 文本长度（整数）
        key: 建议AI保存的关键字
        format 你来举例教ai划分
    returns:
        JSON 格式的分段摘要
    """
    prompt_template = """
    你是一个文本分析AI，需要根据语义连贯性和主题完整性划分段落，并返回JSON格式结果。

    要求：
    1. 按语义自然分块，确保每个块独立完整,不要切断完整的句子，换行符空格均为有效字符
    2. 保留关于的关键信息
    3. 标注字符级起止位置（从0开始）
    {Example}

    请处理以下文本（长度：{length}）：
    {text}
    """
    
    ChatPromptT = ChatPromptTemplate.from_messages([
        ("system", "你是一个文本处理AI，需要按语义划分段落并返回JSON。"),
        ("human", prompt_template)
    ])
    
    return ChatPromptT.invoke({"text": text, "key": key, "length": length,"Example":Example})

def txtsliptPrompt3(text:str,length:int,key:str,format:str=""" { "data": [
    {
      "s": "压缩语义1",
      "b": "段落开始1",
      "e": "段落结束1",
    "r": [在文段中与"e"相同的句子，在文段中与"e"相同的句子....]
    },
    {
      "s": "压缩语义2",
      "b": "段落开始2",
      "e": "段落结束2",
       "r": [在文段中与"e"相同的句子，在文段中与"e"相同的句子....]
    }
.....
  ]
}         
注意e、r里面的信息是帮助段落文本定位的 ，他们的在txt里选取不应该造成歧义"""):
        """
        准确返回段落的开始与结束，使用参数排查结尾的重复次数。对于划分正好在分割处的肯能歧义
        args
                text 被分割的文本
                key  建议ai保存的关键字
                format json格式
        Prompt
        """
        ChatPromptT = ChatPromptTemplate.from_messages([ 
                ("system", "你是一个按照语义划分文本并压缩这些按照语义划分的文本内容的函数，你要按照给出的json格式返回对应的字符内容"),
                ("user",
                 """
                 按一下步骤执行文本任务
                        1按照标点符号、文本语义划分长度为{length}的文本,划分大小你自己确定,每个划分块彼此保留一些重叠的部分。
                        2提出每个文本的:压缩语义与原始文本 （压缩语义保留关键字：{key}）。
                        3按照指定的json格式输出:{format}         
                txt:{text}
                 """ 
                 )
                ])
        return ChatPromptT.invoke({"text": text,"length": length,"key":key,"format":format})  

def txtsliptPrompt(text:str,length:int,key:str,format:str=""" {
  "length": 文本的长度,
  "data": [
    {
      "s": "压缩语义1",
      "b": "段落开始1",
      "e": "段落结束1",
    },
    {
      "s": "压缩语义2",
      "b": "段落开始2",
      "e": "段落结束2",
    }
.....
  ]
}         
注意e里面的信息是帮助段落文本定位的具有唯一性"""):
        """
        准确返回段落的开始与结束，使用参数排查结尾的重复次数。命令ai确保结束标志唯一
        args
                text 被分割的文本
                key  建议ai保存的关键字
                format json格式
        Prompt
        """
        ChatPromptT = ChatPromptTemplate.from_messages([ 
                ("system", "你是一个按照语义划分文本并压缩这些按照语义划分的文本内容的函数，你要按照给出的json格式返回对应的字符内容"),
                ("user",
                 """
                 按一下步骤执行文本任务
                        1按照标点符号、文本语义划分长度为{length}的文本,划分大小你自己确定,每个划分块彼此保留一些重叠的部分。
                        2提出每个文本的:压缩语义与原始文本 （压缩语义保留关键字：{key}）。
                        3按照指定的json格式输出:{format}         
                txt:{text}
                 """ 
                 )
                ])
        return ChatPromptT.invoke({"text": text,"length": length,"key":key,"format":format})  