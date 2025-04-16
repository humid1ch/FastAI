from langchain_core.documents import Document
from langchain_chroma import Chroma

from Global.prompts import image_recognition_prompt
from Global.utils import embeddings, adjust_image, get_image_type, extract_json_raw
from Global.utils import extract_paragraphs
from datetime import datetime


class RAG:
    def __init__(
        self, persist_directory, embeddings, mutil_model: str = "qwen-vl-max-latest"
    ):
        import Global.llm

        self.mutil_llm = Global.llm.mutil_llm(model=mutil_model)
        self.chat_llm = Global.llm.deepseek_by_ds("deepseek-chat")
        self.db = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
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
        message = image_recognition_prompt(image_type, base64_image, image_name)
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
        import Global.prompts

        split_prompt = Global.prompts.text_split_prompt(texts, len(texts), key)

        # json 映射 txt ->Documents
        json_raw = Global.utils.extract_json_raw(
            self.chat_llm.invoke(split_prompt).content
        )
        import json

        try:
            Json = json.loads(json_raw)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError("无效的JSON数据") from e

        texts_splits = extract_paragraphs(texts, json_raw)
        if len(Json["data"]) != len(texts_splits):
            print("len(Json)!=len(texts_splits)")
            return

        documents = []
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

    def storage_documents(self, documents: list[Document] | str, is_async=True):
        if is_async:
            self.db.aadd_documents(documents)
        else:
            self.db.add_documents(documents)

    def query(self, query: str):
        return self.db.similarity_search_with_score(query)
