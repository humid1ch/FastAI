from langchain_core.documents  import Document 
from langchain_community.vectorstores import Chroma


class VDB:
    def __init__(self,persist_directory,embeddings):
        self.db=Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings 
        )
    def storage_split_txt(txts :list[str] | str):
        Document()

    def search(self,query:str):
        pass
    