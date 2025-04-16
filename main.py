from  Global.rag  import *
from  Global.test import *
from  Global.prompts import *
from  Global.llm  import *

json=json斗破苍穹()
rag=RAG("./chroma_db")
rag.storage_json(json,"人物事件",False)
print(rag.query("萧薰儿斗气是几段"))

