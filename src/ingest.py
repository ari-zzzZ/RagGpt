import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import settings
from langchain_core.documents import Document

def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if os.path.isfile(path) and filename.endswith((".txt", ".md")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                try:
                    with open(path, "r", encoding="gbk") as f:
                        text = f.read()
                except Exception as e:
                    print(f" 无法加载文件（编码错误）：{filename}")
                    continue

            docs.append(Document(page_content=text, metadata={"source": path}))
    return docs



def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)


def build_vector_store():
    docs = load_documents(settings.KNOWLEDGE_FOLDER)
    chunks = chunk_documents(docs)
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(settings.INDEX_PATH)
    print(" Vector store built and saved.")