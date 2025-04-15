from  Global.rag  import RAG
from  Global.test import *
from  Global.until import embeddings
txt=txt_斗破苍穹1()
from langchain_huggingface import HuggingFaceEmbeddings 
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",cache_folder="./models",model_kwargs = {'device': 'cuda'})

rag=RAG("./chroma_db",embed)
rag.storage_txt(txt,"人物事件","斗破苍穹第一章",is_Async=False)
print(rag.query("萧薰儿斗气是几段"))