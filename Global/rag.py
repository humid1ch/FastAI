from langchain_core.documents  import Document 
from langchain_chroma import Chroma
from Global.until import embeddings
from Global.until import extract_paragraphs
from datetime import datetime
class RAG:
    def __init__(self,persist_directory,embeddings):
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

    def storage_Document(self,documents : list[Document] | str ,is_Async = True):
        if(is_Async):
            self.db.aadd_documents(documents)
        else:
            self.db.add_documents(documents)

    def query(self,query:str):
        return self.db.similarity_search(query)
    