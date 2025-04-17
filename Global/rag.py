from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from Global.prompts import (
    image_recognition_prompt,
    text_split_prompt,
    json_split_prompt,
    llm_answer_prompt,
)
from Global.utils import (
    adjust_image,
    get_image_type,
    extract_json_raw,
    extract_paragraphs,
)
from Global.llm import mutil_llm, deepseek_by_ds
from datetime import datetime
import json
from json import JSONDecodeError
import regex
import sqlite3

from datetime import datetime
import json  

class RAG:

    def __init__(
        self,
        vector_persist_directory,
        sql_url,
        embeddings,
        mutil_model: str = "qwen-vl-max-latest",
    ):
        # 多模态 视觉大模型
        self.mutil_llm = mutil_llm(model=mutil_model)
        # 聊天大模型
        self.chat_llm = deepseek_by_ds("deepseek-chat")
        # 向量数据库
        self.vector_db = Chroma(
            persist_directory=vector_persist_directory, embedding_function=embeddings
        )
        # 数据库
        self.sql_db = sqlite3.connect(sql_url)
        self.sql_db_cursor = self.sql_db.cursor()
        self.sql_db_cursor.execute(
            """CREATE TABLE IF NOT EXISTS image_raw_data
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                course TEXT NOT NULL,
                ppt_file_page TEXT NOT NULL,
                create_time TEXT NOT NULL,
                title TEXT,
                content TEXT)"""
        )
        self.sql_db.commit()

        self.sql_db_cursor.execute(
            """CREATE TABLE IF NOT EXISTS image_summary_data
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                course TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT,
                date TEXT NOT NULL,
                source TEXT)"""
        )
        self.sql_db.commit()

    def extract_images(self, images_path: list[str]):
        image_infos = []
        for image_path in images_path:
            try:
                image_info = self.extract_image_info(image_path)
                image_infos.append(json.loads(image_info))
            except JSONDecodeError as e:
                # 如果content中存在LaTeX公式, 因为公式中存在各种 \命令
                # 可能会出现异常, json.loads()无法正常加载非法的转义字符
                # 处理方法, 先获取到content, 将原始字符串的content设置为"", 先加载为json数据
                # 然后清理获取到的content, 在赋值给json["content"], 完成json数据的获取
                # 获取 content 值
                content_match = regex.search(
                    r'"context": "(.*?)"(?=\s*,)', image_info, regex.DOTALL
                )
                if content_match:
                    content_value = content_match.group(1)

                # 替换 content 为空字符串
                modified_image_info = regex.sub(
                    r'"context": ".*?"(?=\s*,)',
                    '"context": ""',
                    image_info,
                    flags=regex.DOTALL,
                )
                try:
                    image_info_cur = json.loads(modified_image_info)
                    # 匹配 LaTeX 命令（如 \frac, \mu, \sigma 等）
                    latex_commands = regex.findall(r"\\([a-zA-Z]+)", content_value)
                    # 替换多余的 \，但保留合法的 LaTeX 命令
                    cleaned = regex.sub(
                        r"\\(?![a-zA-Z])", "", content_value
                    )  # 删除无效的 \
                    image_info_cur["context"] = cleaned
                    image_infos.append(image_info_cur)
                except JSONDecodeError as e:
                    print("JSONDecoder Error")
                    continue

        # json数据转字符串
        json_image_infos = json.dumps(image_infos, indent=4, ensure_ascii=False)
        return json_image_infos

    def extract_image_info(self, image_path: str):
        base64_image, image_new_path = adjust_image(
            image_path=image_path, output_path="./adjust_images"
        )
        image_type, image_name = get_image_type(image_new_path)
        # 获取图片提取的提示词
        message = image_recognition_prompt(image_type, base64_image, image_name)
        image_info = self.mutil_llm.invoke(message).content

        try:
            json.loads(image_info)
        except JSONDecodeError as e:
            image_info = extract_json_raw(image_info)

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
        split_prompt = text_split_prompt(texts, len(texts), key)
        # json 映射 txt ->Documents
        json_raw = extract_json_raw(self.chat_llm.invoke(split_prompt).content)
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

            documents.append(
                Document(
                    page_content=Json["data"][i]["s"],
                    metadata={
                        "source": source,
                        "date": date,
                        "rawdata": texts_splits[i],
                    },
                )
            )
        self.storage_documents(documents, is_async)

    def storage_json(self, json_raw: str, key: str, is_async=True, max_retries=3):
        documents = []  # 确保变量初始化
        attempts = 0
        while attempts < max_retries:
            try:
                # 1. 解析输入JSON
                input_json = json.loads(json_raw, strict=False)

                print(f"input_json item count {len(input_json)}")

                if not isinstance(input_json, list):
                    raise ValueError("Input JSON should be a list of segments")

                # 2. 构建完整文本

                full_text = "\n".join(seg.get("context", "") for seg in input_json)
                print(f"full_text {len(full_text)}")

                # 3. 生成分割提示
                prompt = json_split_prompt(
                    json=json_raw, context=full_text, length=len(full_text), key=key
                )

                # 4. 处理LLM响应
                response = self.chat_llm.invoke(prompt)
                ret_json_raw = extract_json_raw(response.content)

                # 5. 解析并验证响应
                ret_json = json.loads(ret_json_raw)
                if "data" not in ret_json or not isinstance(ret_json["data"], list):
                    raise ValueError("Invalid response structure: missing 'data' array")


                # 6. 分割文本并验证长度
                txts_splits = extract_paragraphs(full_text, ret_json_raw)
                if len(ret_json["data"]) != len(txts_splits):
                    raise ValueError(
                        f"Data length mismatch: {len(ret_json['data'])} vs {len(txts_splits)}"
                    )


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
                    # for j in range(len(data_item["src"])):
                    #     documents.append(
                    #         Document(
                    #             page_content=data_item["sum"],
                    #             metadata={
                    #                 "source": data_item["src"][j],
                    #                 "file": data_item["f"][j],
                    #                 "date": data_item["t"][j],
                    #                 "rawdata": txts_splits[i],
                    #             },
                    #         )
                    #     )
                    source_str = ", ".join(data_item["src"])  # 拼接整个 src 列表
                    file_str = ", ".join(data_item["f"])  # 拼接整个 f 列表
                    date_str = ", ".join(data_item["t"])  # 拼接整个 t 列表

                    # 只添加一个 Document
                    documents.append(
                        Document(
                            page_content=data_item["sum"],
                            metadata={
                                "source": source_str,
                                "file": file_str,
                                "date": date_str,
                                "rawdata": txts_splits[i],
                            },
                        )
                    )
                break  # 成功时退出循环

            except json.JSONDecodeError as e:
                print(f"[Attempt {attempts + 1}] JSON解析失败: {str(e)}")
                attempts += 1
            except (KeyError, ValueError, IndexError) as e:
                print(f"[Attempt {attempts + 1}] 数据结构异常: {str(e)}")
                attempts += 1
            except Exception as e:
                print(f"[Attempt {attempts + 1}] 未知错误: {str(e)}")
                attempts += 1

        # 8. 存储文档
        if documents:

            # 原始图片数据, 添加到数据库
            self.save_image_raw_datas_to_db(input_json)
            # 总结图片数据, 添加到数据库
            self.save_image_summary_datas_to_db(documents)
            # 总结图片数据, 添加向量数据量
            self.storage_documents(documents, is_async)
        else:
            print("[Attempt {}] No documents to store.".format(attempts + 1))

    def storage_documents(self, documents: list[Document] | str, is_async=True):
        if is_async:
            self.vector_db.aadd_documents(documents)
        else:
            self.vector_db.add_documents(documents)

    def save_image_raw_datas_to_db(self, image_raw_datas: list):
        for image_data in image_raw_datas:
            course = "固定课程"
            ppt_file_page = image_data["source"]
            create_time = image_data["date"]
            title = image_data["title"]
            content = image_data["context"]
            self.insert_image_raw_data(
                course, ppt_file_page, create_time, title, content
            )

    def save_image_summary_datas_to_db(self, image_raw_datas: list[Document]):
        for image_data in image_raw_datas:
            course = "固定课程"
            summary = image_data.page_content
            content = image_data.metadata["rawdata"]
            source = image_data.metadata["source"]
            date = image_data.metadata["date"]
            self.insert_image_summary_data(course, summary, date, content, source)

    def insert_image_raw_data(
        self,
        course: str,
        ppt_file_page: str,
        create_time: str,
        title: str = None,
        content: str = None,
    ):
        """
        向 image_raw_data 表中插入数据

        参数:
            course: 课程名称
            ppt_file_page: PPT文件页码
            create_time: 创建时间
            title: 标题(可选)
            content: 内容(可选)
        """
        try:
            self.sql_db_cursor.execute(
                """INSERT INTO image_raw_data 
                   (course, ppt_file_page, create_time, title, content) 
                   VALUES (?, ?, ?, ?, ?)""",
                (course, ppt_file_page, create_time, title, content),
            )
            self.sql_db.commit()
            print("成功插入 image_raw_data 数据")
        except Exception as e:
            print(f"插入 image_raw_data 数据失败: {e}")
            self.sql_db.rollback()

    def insert_image_summary_data(
        self,
        course: str,
        summary: str,
        date: str,
        content: str = None,
        source: str = None,
    ):
        """
        向 image_summary_data 表中插入数据

        参数:
            course: 课程名称
            summary: 摘要内容
            content: 详细内容(可选)
            source: 来源(可选)
        """
        try:
            self.sql_db_cursor.execute(
                """INSERT INTO image_summary_data 
                   (course, summary, content, date, source) 
                   VALUES (?, ?, ?, ?, ?)""",
                (course, summary, content, date, source),
            )
            self.sql_db.commit()
            print("成功插入 image_summary_data 数据")
        except Exception as e:
            print(f"插入 image_summary_data 数据失败: {e}")
            self.sql_db.rollback()

    def query(self, query: str):
        return self.vector_db.similarity_search_with_score(query, k=10)

    def answer(self, query: str):
        retriever = RunnableLambda(self.query)

        answer_prompt = llm_answer_prompt()

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | answer_prompt
            | self.chat_llm
            | StrOutputParser()
        )

        result = chain.invoke(query)
        print(result)
