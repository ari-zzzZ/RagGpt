import os
import fitz
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import settings

def load_documents(folder_path):
    docs = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path = os.path.join(root, filename)
            lower = filename.lower()
            if lower.endswith((".txt", ".md")):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(path, "r", encoding="gbk", errors="ignore") as f:
                        text = f.read()
                docs.append(Document(page_content=text, metadata={"source": path}))
            elif lower.endswith(".pdf"):
                try:
                    text = ""
                    pdf = fitz.open(path)
                    for page in pdf:
                        text += page.get_text()
                    docs.append(Document(page_content=text, metadata={"source": path}))
                except Exception as e:
                    print(f"无法读取 PDF：{filename}，错误：{e}")
    return docs

def build_vector_store():
    """
    构建或增量更新 FAISS 向量索引：
    - 首次运行时创建索引目录（settings.INDEX_PATH）
    - 再次运行时仅对新增文件进行 embedding 并追加
    注意：settings.INDEX_PATH 应配置为索引目录，而非单个文件路径
    """
    '''embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)'''
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    # 确保索引目录存在或记录为空
    if os.path.isdir(settings.INDEX_PATH):
        db = FAISS.load_local(settings.INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        existing_sources = {doc.metadata.get("source") for doc in db.docstore._dict.values()}
    else:
        db = None
        existing_sources = set()

    docs = load_documents(settings.KNOWLEDGE_FOLDER)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    new_files = 0
    new_chunks = 0
    for doc in docs:
        src = doc.metadata.get("source")
        print(f"[Building] 处理文件：{src}")
        if src in existing_sources:
            continue
        chunks = splitter.split_documents([doc])
        new_files += 1
        new_chunks += len(chunks)
        if db is None:
            db = FAISS.from_documents(chunks, embeddings)
        else:
            db.add_documents(chunks)

    # 保存到目录
    os.makedirs(settings.INDEX_PATH, exist_ok=True)
    if db is not None:
        db.save_local(settings.INDEX_PATH)

    print(f"增量索引完成：新增文档 {new_files} 个，共新增 chunks {new_chunks} 个。")