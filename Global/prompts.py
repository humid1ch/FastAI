def image_prompt_by_url(image_url: str, query: str):
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

    from langchain_core.prompts import ChatPromptTemplate
    ChatPromptT = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "user",
                [
                    {"type": "image_url", "image_url": {"url": f"{image_url}"}},
                    {"type": "text", "text": "{query}"},
                ],
            ),
        ]
    )
    return ChatPromptT.invoke({"query": query})


def image_prompt_by_path(path: str, query: str, model: str = "qwen-vl-max"):
    """
    自动编码传输图片文件

    args
            path 路径
            query user的提示词
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
    import Global.utils as imagedata
    from langchain_core.prompts import ChatPromptTemplate
    image_data, image_path = imagedata.adjust_image(path, model)
    ChatPromptT = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                    {"type": "text", "text": "{query}"},
                ],
            ),
        ]
    )
    return ChatPromptT.invoke({"query": query})


def image_recognition_prompt(image_type, base64_image, image_name):
    from langchain_core.messages import SystemMessage
    from langchain_core.messages import HumanMessage
    return [
        SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are an assistant who understands and analyzes image text. You can return a Json string in a certain format according to the user's requirements.",
                }
            ]
        ),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_type};base64,{base64_image}"},
                },
                {
                    "type": "text",
                    "text": f"""
                        这是大学课堂上PPT文件的一张截图。图片的文件名为{image_name}, 截图时间为 "2025-01-01 13:44:38"。请理解并分析此图片，并提供以下信息：
                        1. 截图的文件名
                        2. 图片所属的PPT文件名及页码, 即截图的实际来源
                        3. 截图的时间
                        4. PPT标题
                        5. 图片中完整、详细、条理清晰、逻辑清晰的文本内容, 不要做不必要的增删改
                        要求：
                        1. 所有输出均以绝对纯净的原始JSON字符串格式输出。输出内容禁止以任何形式的代码块进行包装。
                        2. 图片的文件名, 在"filename"键的值中描述。
                        2. 图片中所述的PPT标题, 在"title"键的值中描述, 如果图片本身不存在标题, 结合图片内容自拟一个
                        3. 图片中的完整详细内容, 在“context”键的值中描述。
                        4. 图片的所属来源(PPT文件名及页码), 在"source"键的值中描述。
                        5. 截图的时间在"date"键的值中描述。
                        6. 如果图片中存在数学公式, 请将公式输出为完整有效的 LaTeX 语法格式
                        即完整的输出格式如下：
                        '{{
                            "filename": "不断促进全体人民共同富裕_pptx_第三页.jpeg",
                            "title": "不断促进全体人民共同富裕"
                            "context" : "xxxxxxxxx",
                            "source": "不断促进全体人民共同富裕_pptx_第3页",
                            "date": "2025-01-01 13:44:38"
                        }}'
                        """,
                },
            ]
        ),
    ]


def text_split_prompt1(
    text: str,
    length: int,
    key: str,
    format: str = """
    {
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
    }
    """,
):
    """
    ai按ai选定的固定大小划分
    args
            text 被分割的文本
            key  建议ai保存的关键字
            format json格式        被修改就要重新测试
    Prompt
    """
    from langchain_core.prompts import ChatPromptTemplate
    ChatPromptT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个按照语义划分文本并压缩这些按照语义划分的文本内容的函数，你要按照给出的json格式返回对应的字符内容",
            ),
            (
                "user",
                """
                 按一下步骤执行文本任务
                        1按照标点符号、文本语义划分长度为{length}的文本,划分大小你自己确定,每个划分块彼此保留一些重叠的部分。
                        2提出每个文本的:压缩语义与原始文本 （压缩语义保留关键字：{key}）。
                        3按照指定的json格式输出:{format}         
                txt:{text}
                 """,
            ),
        ]
    )
    return ChatPromptT.invoke(
        {"text": text, "length": length, "key": key, "format": format}
    )


def text_split_prompt2(
    text: str,
    length: int,
    key: str,
    example: str = """
    4. 返回格式必须严格遵循：
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
    """,
):
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
    这是示例: {example}

    请处理以下文本:
    文本长度: {length}
    文本内容: {text}
    """
    from langchain_core.prompts import ChatPromptTemplate
    ChatPromptT = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个文本处理AI，需要按语义划分段落并返回JSON。"),
            ("human", prompt_template),
        ]
    )

    return ChatPromptT.invoke(
        {"text": text, "key": key, "length": length, "example": example}
    )


def text_split_prompt3(
    text: str,
    length: int,
    key: str,
    format: str = """
    {
        "data": [
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
    注意e、r里面的信息是帮助段落文本定位的 ，他们的在txt里选取不应该造成歧义
    """,
):
    """
    准确返回段落的开始与结束，使用参数排查结尾的重复次数。对于划分正好在分割处的肯能歧义
    args
            text 被分割的文本
            key  建议ai保存的关键字
            format json格式
    Prompt
    """
    from langchain_core.prompts import ChatPromptTemplate
    ChatPromptT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个按照语义划分文本并压缩这些按照语义划分的文本内容的函数，你要按照给出的json格式返回对应的字符内容",
            ),
            (
                "user",
                """
                 按一下步骤执行文本任务
                        1按照标点符号、文本语义划分长度为{length}的文本,划分大小你自己确定。
                        2提出每个文本的:压缩语义与原始文本 （压缩语义保留关键字：{key}）。
                        3按照指定的json格式输出:{format}         
                txt:{text}
                 """,
            ),
        ]
    )
    return ChatPromptT.invoke(
        {"text": text, "length": length, "key": key, "format": format}
    )


def text_split_prompt(
    text: str,
    length: int,
    key: str,
    format: str = """
    {
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
            ...
        ]
    }         
    注意e里面的信息是帮助段落文本定位的具有唯一性
    """,
):
    """
    准确返回段落的开始与结束，使用参数排查结尾的重复次数。命令ai确保结束标志唯一
    args
            text 被分割的文本
            key  建议ai保存的关键字
            format json格式
    Prompt
    """
    from langchain_core.prompts import ChatPromptTemplate
    ChatPromptT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个按照语义划分文本, 并压缩这些按照语义划分后文本内容的助手，你要按照给出的json格式返回对应的字符内容",
            ),
            (
                "user",
                """
                按以下步骤执行文本任务
                       1. 按照标点符号、文本语义划分 总长度为 {length} 的一段文本, 每块划分长度你自己确定, 每个划分块彼此保留一些有效的重叠部分。
                       2. 提取出每个文本的: 压缩语义与原始文本, 要求压缩语义保留关键字: {key}
                       3. 按照指定的json格式输出, 格式为 {format}         
                需要划分的文本为: {text}
                """,
            ),
        ]
    )
    return ChatPromptT.invoke(
        {"text": text, "length": length, "key": key, "format": format}
    )

def json_split_prompt(
    json: str,
    context: str,
    length: int,
    key: str,
    format: str = """
     示例
    输入的完整json为:
    [
        {
            "filename": "yyy.png",
            "context": "在一片古老的森林中，阳光透过茂密的枝叶洒下，形成一片片光斑。少年林羽手持长剑，眼神坚定地穿梭在树林间。他是来参加森林试炼的，希望能在这次试炼中证明自己。忽然，一只凶猛的野猪从草丛中窜出，朝着林羽冲了过来。林羽迅速侧身闪过，然后挥剑刺向野猪。野猪发出一声惨叫，倒在地上。然而，这只野猪只是开始，更多的野兽闻声赶来。林羽陷入了苦战，汗水湿透了他的衣衫，但他始终没有退缩。周围的人看到他的坚持，有人开始嘲笑：‘就他这样，还想通过试炼，简直是异想天开。’‘估计撑不了多久就得被抬出去了。’林羽听到这些嘲笑，心中有些失落，但他咬了咬牙，更加奋力地战斗着。就在他快要坚持不住的时候，一个神秘的老者出现了。老者轻轻一挥衣袖，那些野兽便纷纷退去。林羽感激地看着老者，老者微笑着说：‘孩子，坚持就是胜利，不要被他人的话语所影响。’随后，老者消失在了树林中。试炼继续，‘下一个，苏瑶！’",
            "source": "《森林传奇》第 5 页",
            "date": "2025-11-15"
        },
        {
            "filename": "yyy2.png",
            "context": "陈轩大步走上试炼场地，他身材魁梧，气势逼人。他手中紧握着一把巨斧，用力一挥，便砍倒了一棵大树。‘好厉害的力量！’众人纷纷喝彩。然而，就在他得意之时，一只巨大的熊怪从树林深处冲了出来。熊怪力大无穷，陈轩与它激烈战斗，一时间难分胜负。",
            "source": "《森林传奇》第 6 页",
            "date": "2025-11-15"
        } ,
        {
            "filename": "yyy3.png",
            "context": "苏瑶在一旁看着，心中有些担忧。她虽然决定专注试炼，但看到朋友有危险，还是忍不住出手相助。她施展自然之力，束缚住了熊怪的行动。陈轩趁机一斧砍向熊怪，将其击败。陈轩感激地看向苏瑶，苏瑶微笑着说：‘朋友之间，不必言谢。’这时，林羽也来到了他们身边，他的伤势已经在神秘老者的帮助下有所恢复。林羽看着苏瑶和陈轩，心中涌起一股温暖。‘接下来，让我们一起面对剩下的试炼吧！’林羽坚定地说。",
            "source": "《森林传奇》第 7 页",
            "date": "2025-11-15"
        }
    ]
    要划分的"context"为:
在一片古老的森林中，阳光透过茂密的枝叶洒下，形成一片片光斑。少年林羽手持长剑，眼神坚定地穿梭在树林间。他是来参加森林试炼的，希望能在这次试炼中证明自己。忽然，一只凶猛的野猪从草丛中窜出，朝着林羽冲了过来。林羽迅速侧身闪过，然后挥剑刺向野猪。野猪发出一声惨叫，倒在地上。然而，这只野猪只是开始，更多的野兽闻声赶来。林羽陷入了苦战，汗水湿透了他的衣衫，但他始终没有退缩。周围的人看到他的坚持，有人开始嘲笑：‘就他这样，还想通过试炼，简直是异想天开。’‘估计撑不了多久就得被抬出去了。’林羽听到这些嘲笑，心中有些失落，但他咬了咬牙，更加奋力地战斗着。就在他快要坚持不住的时候，一个神秘的老者出现了。老者轻轻一挥衣袖，那些野兽便纷纷退去。林羽感激地看着老者，老者微笑着说：‘孩子，坚持就是胜利，不要被他人的话语所影响。’随后，老者消失在了树林中。试炼继续，‘下一个，苏瑶！’
陈轩大步走上试炼场地，他身材魁梧，气势逼人。他手中紧握着一把巨斧，用力一挥，便砍倒了一棵大树。‘好厉害的力量！’众人纷纷喝彩。然而，就在他得意之时，一只巨大的熊怪从树林深处冲了出来。熊怪力大无穷，陈轩与它激烈战斗，一时间难分胜负。
苏瑶在一旁看着，心中有些担忧。她虽然决定专注试炼，但看到朋友有危险，还是忍不住出手相助。她施展自然之力，束缚住了熊怪的行动。陈轩趁机一斧砍向熊怪，将其击败。陈轩感激地看向苏瑶，苏瑶微笑着说：‘朋友之间，不必言谢。’这时，林羽也来到了他们身边，他的伤势已经在神秘老者的帮助下有所恢复。林羽看着苏瑶和陈轩，心中涌起一股温暖。‘接下来，让我们一起面对剩下的试炼吧！’林羽坚定地说。
即, 要划分的"context"来源是完整json数组中的"context"字段
    输出 内容为:
    {
        "data": [
            {
                "sum": "林羽森林试炼遇兽苦战遭嘲笑，神秘老者相助",
                "b": "在一片古老的森林中，阳光透过茂密的枝叶洒下，形成一片片光斑。",
                "e": "‘下一个，苏瑶！’",
                "f": ["yyy.png"],
                "src": ["《森林传奇》第 5 页"],
                "t": ["2025-11-15"]
            },
            {
                "sum": "陈轩战斗遇险，苏瑶相助，三人决定共迎试炼",
                "b": "陈轩大步走上试炼场地，他身材魁梧，气势逼人。",
                "e": "‘接下来，让我们一起面对剩下的试炼吧！’",
                "f": ["yyy2.png","yyy3.png"],
                "src": ["《森林传奇》第 6 页","《森林传奇》第 7 页"],
                "t": ["2025-11-15","2025-11-15"]
            }
        ]
    }
   如果摘要总结只有一个元素, 也要放入"data"数组中:
    {
        "data": [
            {
                "sum": "陈轩战斗遇险，苏瑶相助，三人决定共迎试炼",
                "b": "陈轩大步走上试炼场地，他身材魁梧，气势逼人。",
                "e": "‘接下来，让我们一起面对剩下的试炼吧！’",
                "f": ["yyy2.png","yyy3.png"],
                "src": ["《森林传奇》第 6 页","《森林传奇》第 7 页"],
                "t": ["2025-11-15","2025-11-15"]
            }
        ]
    }
    """,
):
    """
    args
        json格式的文本
        [
            {
                "filename":"1",
                "context":"1",
                "source":"1",
                "date" : "1"
            },
            {
                "filename":"2",
                "context":"2",
                "source":"2",
                "date" : "2"
            }
        ]
        key  建议ai保存的关键字
        format json格式
        Prompt
    """
    from langchain_core.prompts import ChatPromptTemplate
    ChatPromptT = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                """
                按照以下步骤处理"context"文本:
                1. 铭记一个前提: "context"文本来自于json串中的"context"键值, json串中, "filename"表示"context"所在的文件名, "source"表示"context"的实际来源, "date"表示文件创建的时间
                2. 理解文本, 并按照上下文关联以及语义划分出若干文本块, 并记录: 划分出的文本块的开头句"b"和结尾句"e", 划分出的文本块的来源文件名以及划分出文本块的实际来源, 这个文本块可能横跨多个json的"context", 所以文本块可能拥有多个文件名和实际来源
                3. 对划分出每个文本块做摘要总结, 保留关键词"{key}"
                4. 每个文本块的信息作为一个json元素: 摘要总结描述在"sum"键的值中, 开头句描述在"b"键的值中, 结尾句描述在"e"键的值中, 来源文件描述在"f"键的值中(可能是一个数组), 实际来源描述在"src"键的值中(可能是一个数组), 来源文件的时间描述在"t"键的值中(可能是一个数组)
                5. 整合每个文本块的json元素作为一个json数组输出, 数组名为"data"
                6. 即使只有一个文本块json元素, 也要放入数组名为"data"的数组中, 输出总是包含"data"数组
                7. 可以参考示例: {format}
                8. 需要处理的"context"文本: {context}, 总长度为:{length}
                9. "context"文本的原json数组: {json}
                """,
            ),
        ]
    )
    return ChatPromptT.invoke(
        {
            "json": json,
            "length": length,
            "context": context,
            "key": key,
            "format": format,
        }
    )


def llm_answer_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_template(
        template="""
        仅根据以下上下文回答问题，其中上下文为：{context}。
        附加要求：
        1. 所有输出均以绝对纯正的原始 JSON 字符串格式输出。输出内容禁止以任何形式的代码块进行封装。
        2. 问题的答案在“answer”键的值中描述。
        3. “answer”键的值不应包含诸如“基于上下文”、“基于附加信息”等内容。
        4. “answer”键的值不应指明答案中引用数据的来源。
        5. 回答问题后，使用单独的“source”键的值描述答案中引用数据的完整的、所有的来源, 即使引用数据重复（必须包含文件名和页码）。
        问题：{question}
        """
    )
