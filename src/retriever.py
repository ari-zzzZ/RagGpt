import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import settings
from pathlib import Path

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
    #最轻量／最常用的接口，直接拿到检索到的文本片段列表，方便你拼接、展示或评估
    topk = k or settings.TOP_K
    db = _load_db()
    docs = db.similarity_search(query, k=topk)
    return [doc.page_content for doc in docs]


def get_topk_docs(query: str, k: int = None):
    """返回带完整 metadata 和 score 的 Document 对象列表"""
    #当需要访问原始的 metadata（比如 source 路径）时，用它可以拿到上下文来源信息。
    topk = k or settings.TOP_K
    db = _load_db()
    return db.similarity_search(query, k=topk)

def get_topk_docs_with_score(query: str, k: int = None):
    #当你不仅要上下文，还想拿到 FAISS 返回的相似度分数时，用它最合适
    topk = k or settings.TOP_K
    db = _load_db()
    docs_and_scores = db.similarity_search_with_score(query, k=topk)

    # 给 metadata 增加 id 字段（取 source 文件名去掉后缀）
    patched = []
    for doc, score in docs_and_scores:
        src = doc.metadata.get("source", "")
        doc.metadata["id"] = Path(src).stem
        patched.append((doc, score))

    return patched

