from langchain_core.documents import Document
from langchain_chroma import Chroma

from Global import prompts, llm  
from Global.utils import default_embedding, adjust_image, get_image_type, extract_json_raw,extract_paragraphs

from datetime import datetime
import json  

class RAG:
    def __init__(self,persist_directory,mutil_model:str = "qwen-vl-max-latest",embeddings=None):
        self.mutil_llm = llm.mutil_llm(model=mutil_model)
        self.chat_llm =  llm.deepseek_by_ds("deepseek-chat")
        #不传入embeddings时，使用默认的embeddings
        if embeddings==None: 
            self.db=Chroma(
            persist_directory=persist_directory,
            embedding_function=default_embedding() 
            )
        else:
            self.db=Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
            )

        
    
    def extract_images(self, images_path: list[str]):
        import json
        from json import JSONDecodeError

        image_infos = []
        for image_path in images_path:
            try:
                image_info = self.extract_image_info(image_path)
                image_infos.append(json.loads(image_info))
            except JSONDecodeError as e:
                import regex
                # 如果content中存在LaTeX公式, 因为公式中存在各种 \命令
                # 可能会出现异常, json.loads()无法正常加载非法的转义字符
                # 处理方法, 先获取到content, 将原始字符串的content设置为"", 先加载为json数据
                # 然后清理获取到的content, 在赋值给json["content"], 完成json数据的获取
                content_match = regex.search(r'"content": "(.*?)"(?=\s*})', image_info, regex.DOTALL)
                if content_match:
                    content_value = content_match.group(1)
                # 替换 content 为空字符串
                modified_image_info = regex.sub(
                    r'"content": ".*?"(?=\s*})',
                    '"content": ""',
                    image_info,
                    flags=regex.DOTALL
                )
                try:
                    image_info_cur = json.loads(modified_image_info)
                    # 匹配 LaTeX 命令（如 \frac, \mu, \sigma 等）
                    latex_commands = regex.findall(r'\\([a-zA-Z]+)', content_value)
                    # 替换多余的 \，但保留合法的 LaTeX 命令
                    cleaned = regex.sub(r'\\(?![a-zA-Z])', '', content_value)  # 删除无效的 \
                    image_info_cur["content"] = cleaned
                    image_infos.append(image_info_cur)
                except JSONDecodeError as e:
                    print("JSONDecoder Error")
                    return ""

        json_image_infos = json.dumps(image_infos, indent=4, ensure_ascii=False)
        return json_image_infos

    def extract_image_info(self, image_path: str):
        base64_image, image_new_path = adjust_image(image_path=image_path, output_path="./adjust_images")
        image_type, image_name = get_image_type(image_new_path)
        print(
            f"image_new_path: {image_new_path}, image_type: {image_type}, image_name: {image_name}"
        )
        message = prompts.image_recognition_prompt(image_type, base64_image, image_name)
        image_info = extract_json_raw(self.mutil_llm.invoke(message).content)

        return image_info
    def storage_txt(
        self,
        texts: str,
        key: str,
        source: dict,
        date=datetime.now().strftime("%Y-%m-%d"),
        is_async=True,
    ):
        # ai分割
        split_prompt = prompts.text_split_prompt(texts, len(texts), key)

        # json 映射 txt ->Documents
        json_raw = extract_json_raw(
            self.chat_llm.invoke(split_prompt).content
        )

        try:
            Json = json.loads(json_raw)
        except:
            print("返回的json 解析出错")
            return 
        txts_splits=extract_paragraphs(texts,json_raw)
        if len(Json["data"])!=len(txts_splits):
            print("len(Json)!=len(txts_splits)")
            return 
        Documents=[]
        for i in range(len(Json["data"])):
            Documents.append(Document(page_content=Json["data"][i]["s"], metadata={"source": source, "date": date,"rawdata":txts_splits[i]}))     
        self.storage_Document(Documents,is_async)
    def storage_json(self, json_raw: str, key: str, is_async=True, max_retries=3):
        documents = []  # 确保变量初始化
        attempts = 0
        while attempts < max_retries:
            try:
                # 1. 解析输入JSON
                input_json = json.loads(json_raw, strict=False)
                if not isinstance(input_json, list):
                    raise ValueError("Input JSON should be a list of segments")

                # 2. 构建完整文本
                full_text = "".join(seg.get("context", "") for seg in input_json)
                
                # 3. 生成分割提示
                prompt = prompts.json_slipt_prompt(json_raw, len(full_text), key)
                llm_instance = llm.deepseek_by_ds("deepseek-chat")  # 统一模型名称
                
                # 4. 处理LLM响应
                response = llm_instance.invoke(prompt)
                ret_json_raw = extract_json_raw(response.content)
                
                # 5. 解析并验证响应
                ret_json = json.loads(ret_json_raw)
                if "data" not in ret_json or not isinstance(ret_json["data"], list):
                    raise ValueError("Invalid response structure: missing 'data' array")
                    
                # 6. 分割文本并验证长度
                txts_splits = extract_paragraphs(full_text, ret_json_raw)
                if len(ret_json["data"]) != len(txts_splits):
                    raise ValueError(f"Data length mismatch: {len(ret_json['data'])} vs {len(txts_splits)}")

                # 7. 构建文档集合
                documents = []
                for i, data_item in enumerate(ret_json["data"]):
                    # 验证元数据字段
                    required_keys = {"sum", "src", "f", "t"}
                    if not all(k in data_item for k in required_keys):
                        raise KeyError(f"Missing required keys in data item {i}")
                        
                    # 验证数组长度一致性
                    metadata_fields = [data_item["src"], data_item["f"], data_item["t"]]
                    if len(set(map(len, metadata_fields))) != 1:
                        raise ValueError(f"Metadata array lengths mismatch in item {i}")
                    
                    # 构建文档
                    for j in range(len(data_item["src"])):
                        documents.append(Document(
                            page_content=data_item["sum"],
                            metadata={
                                "source": data_item["src"][j],
                                "file": data_item["f"][j],
                                "date": data_item["t"][j],
                                "rawdata": txts_splits[i]
                            }
                        ))
                break  # 成功时退出循环
                
            except json.JSONDecodeError as e:
                print(f"[Attempt {attempts+1}] JSON解析失败: {str(e)}")
                attempts += 1
            except (KeyError, ValueError, IndexError) as e:
                print(f"[Attempt {attempts+1}] 数据结构异常: {str(e)}")
                attempts += 1
            except Exception as e:
                print(f"[Attempt {attempts+1}] 未知错误: {str(e)}")
                attempts += 1

        # 8. 存储文档
        if documents:
            self.storage_document(documents, is_async)
        else:
            print("[Attempt {}] No documents to store.".format(attempts+1))
    def storage_document(self,documents : list[Document] | str ,is_async = True):
        if(is_async):
            self.db.aadd_documents(documents)
        else:
            self.db.add_documents(documents)
    def query(self,query:str):
        return self.db.similarity_search(query)
    
    