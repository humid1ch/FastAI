import json
import os
from langchain_community.embeddings import DashScopeEmbeddings

from  Global.rag  import RAG
from  Global.utils import load_images

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key="sk-1a9b0010300a4ff6b7b909a00754ea7c"
)

rag = RAG(persist_directory="./chroma_image_db", embeddings=embeddings)

current_dir = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(current_dir, "images")
all_images = load_images(images_path)

# while True:
images_infos = rag.extract_images(all_images)
print(images_infos)
image_infos_json = json.loads(images_infos)
print(image_infos_json)

