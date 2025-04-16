import json
import os

from langchain_community.embeddings import DashScopeEmbeddings

from  Global.rag  import RAG
from  Global.test import *
from Global.utils import load_images

# txt=txt_斗破苍穹1()
# from langchain_huggingface import HuggingFaceEmbeddings
# embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",cache_folder="./models")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key="sk-1a9b0010300a4ff6b7b909a00754ea7c"
)

# rag=RAG("./chroma_db", embed)
# rag.storage_txt(txt,"人物事件","斗破苍穹第一章", is_async=False)
# print(rag.query("萧薰儿斗气是几段"))

rag = RAG(persist_directory="./chroma_image_db", embeddings=embeddings)

current_dir = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(current_dir, "images")
all_images = load_images(images_path)

# # while True:
images_infos = rag.extract_images(all_images)
print(images_infos)
rag.storage_json(json_raw=images_infos, key="", is_async=False)

print(rag.query("什么是共同富裕? 共同富裕的内涵是什么?"))
