import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# 查看所有集合
collections = client.list_collections()
for collection in collections:
    print(collection)
    
# 获取集合统计信息
collection = client.get_collection("your_collection_name")
print(f"Count: {collection.count()}")