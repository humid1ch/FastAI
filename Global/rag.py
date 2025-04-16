from langchain_core.documents  import Document 
from langchain_chroma import Chroma

from Global.until import defaultembeddings, extract_json_raw,extract_paragraphs
from Global import prompts, llm  

from datetime import datetime
import json  

class RAG:
    def __init__(self,persist_directory,embeddings=None):
        
        #不传入embeddings时，使用默认的embeddings
        if embeddings==None: 
            self.db=Chroma(
            persist_directory=persist_directory,
            embedding_function=defaultembeddings() 
            )
        else:
            self.db=Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
            )

        
    def storage_txt(self,txts : str,key:str,source:dict,date=datetime.now().strftime("%Y-%m-%d"),is_Async = True):
        #ai分割
        import Global.prompts
        Prompt=Global.prompts.txtsliptPrompt(txts,len(txts),key)
        import Global.llm
        llm=Global.llm.deepseek_by_ds("deepseek-chat")
        #json 映射 txt ->Documents
        json_raw=Global.until.extract_json_raw(llm.invoke(Prompt).content)
        import json
        try:
            Json = json.loads(json_raw)
        except:
            print("返回的json 解析出错")
            return 
        txts_splits=extract_paragraphs(txts,json_raw)
        if len(Json["data"])!=len(txts_splits):
            print("len(Json)!=len(txts_splits)")
            return 
        Documents=[]
        for i in range(len(Json["data"])):
            Documents.append(Document(page_content=Json["data"][i]["s"], metadata={"source": source, "date": date,"rawdata":txts_splits[i]}))     
        self.storage_Document(Documents,is_Async)
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
                prompt = prompts.jsonliptPrompt(json_raw, len(full_text), key)
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
            self.storage_Document(documents, is_async)
        else:
            print("[Attempt {}] No documents to store.".format(attempts+1))
    def storage_Document(self,documents : list[Document] | str ,is_async = True):
        if(is_async):
            self.db.aadd_documents(documents)
        else:
            self.db.add_documents(documents)
    def query(self,query:str):
        return self.db.similarity_search(query)
    