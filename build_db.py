import json
import chromadb

print("正在初始化本地向量数据库...")
# 创建一个本地文件夹 ./chem_db 来永久保存数据库
client = chromadb.PersistentClient(path="./chem_db")

# 创建一个名为 "reactions" 的集合
collection = client.get_or_create_collection(name="reactions")

# 读取我们刚刚写的 json 数据
with open("reaction_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 准备存入数据库的格式
documents = [] # 给 AI 搜索用的文本（比如：罗宾逊环化 环己酮）
metadatas = [] # 背后暗藏的绝对正确的 SMILES
ids = []       # 唯一编号

for item in data:
    documents.append(item["keywords"])
    # 将 smiles_list 转成字符串存起来
    metadatas.append({"name": item["name"], "smiles_list": json.dumps(item["smiles_list"])})
    ids.append(item["id"])

# 写入数据库
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("✅ 数据注入成功！你的 Agent 现在拥有了不会忘的真理记忆。")