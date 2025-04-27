import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import settings

# 延迟加载
_db = None
_embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

def _load_db():
    global _db
    if _db is None:
        if not os.path.isdir(settings.INDEX_PATH):
            raise RuntimeError(f"FAISS 索引目录不存在，请先运行 'build-index'：{settings.INDEX_PATH}")
        _db = FAISS.load_local(
            settings.INDEX_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
    return _db


def get_relevant_chunks(query: str, k: int = None):
    """返回最相关的文本 chunk 列表"""
    topk = k or settings.TOP_K
    db = _load_db()
    docs = db.similarity_search(query, k=topk)
    return [doc.page_content for doc in docs]


def get_topk_docs(query: str, k: int = None):
    """返回带完整 metadata 和 score 的 Document 对象列表"""
    topk = k or settings.TOP_K
    db = _load_db()
    return db.similarity_search(query, k=topk)
