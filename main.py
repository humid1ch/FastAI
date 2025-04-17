import json
import os
from langchain_community.embeddings import DashScopeEmbeddings

from Global.rag import RAG
from Global.test import *
from Global.utils import load_images

# txt=txt_斗破苍穹1()
# from langchain_huggingface import HuggingFaceEmbeddings
# embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",cache_folder="./models")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3", dashscope_api_key="sk-8f1750138c094448a23c3cca28c3acce"
)


# rag=RAG("./chroma_db", embed)
# rag.storage_txt(txt,"人物事件","斗破苍穹第一章", is_async=False)
# print(rag.query("萧薰儿斗气是几段"))

rag = RAG(
    vector_persist_directory="./chroma_image_db",
    sql_url="./sqlite_db.sqlite3",
    embeddings=embeddings,
)

# current_dir = os.path.dirname(os.path.abspath(__file__))
# images_path = os.path.join(current_dir, "images")
# all_images = load_images(images_path)
# images_infos = rag.extract_images(all_images)
#
# # 使用 with 语句自动管理文件关闭
# # with open("./打造数字经济新优势.txt", "r", encoding="utf-8") as file:
# #     images_infos = file.read()  # 读取全部内容到字符串变量 text
# print(images_infos)
#
# rag.storage_json(json_raw=images_infos, key="", is_async=False)

print(rag.query("什么是共同富裕? 共同富裕的内涵是什么?"))
print(rag.query("社会主义最大的优越性是什么?"))
print(rag.query("什么是排序?"))
print(rag.query("什么是数字经济?"))

print(rag.answer("什么是共同富裕? 共同富裕的内涵是什么?"))
print(rag.answer("社会主义最大的优越性是什么?"))
print(rag.answer("什么是排序?"))
print(rag.answer("什么是数字经济?"))
print(rag.answer("不定积分的性质是什么?"))
print(rag.answer("原函数与不定积分的概念是什么?定义是什么"))

