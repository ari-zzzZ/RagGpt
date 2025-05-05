import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import settings
from pathlib import Path
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# 延迟加载
_db = None
_embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
_reranker = None
_bm25 = None
_bm25_docs = None


def _load_db():
    #懒加载 FAISS 向量库
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

def _load_bm25():
    """
    懒加载 BM25 索引，初步稀疏检索
    """
    global _bm25, _bm25_docs
    if _bm25 is None:
        db = _load_db()
        # 从 FAISS docstore 中获取所有文档
        _bm25_docs = list(db.docstore._dict.values())
        # 简单 tokenizer: 按空白拆分
        texts = [doc.page_content.split() for doc in _bm25_docs]
        _bm25 = BM25Okapi(texts)
    return _bm25, _bm25_docs

def _load_reranker():
    """
    懒加载 CrossEncoder reranker
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker


def get_topk_docs(query: str, k: int = None):
    """
    三阶段检索：BM25 → Dense → Cross-Encoder 重排序

    :param query: 用户查询
    :param k: 最终需要的 top k 数量
    :return: List[Document]
    """
    topk = k or settings.TOP_K
    db = _load_db()
    # 阶段1: BM25 初筛
    bm25_k = topk * 20
    bm25, docs = _load_bm25()
    tokenized = query.split()
    scores = bm25.get_scores(tokenized)
    # 取 BM25 top-n
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bm25_k]
    bm25_docs = [docs[i] for i in idxs]

    # 阶段2: Dense 向量检索
    dense_k = topk * 3
    dense_docs = db.similarity_search(query, k=dense_k)

    # 合并去重
    unique = {}
    for doc in bm25_docs + dense_docs:
        src = doc.metadata.get('source') or doc.metadata.get('id')
        unique[src] = doc
    candidates = list(unique.values())
    if not candidates:
        return []

    # 阶段3: Cross-Encoder 重排序
    reranker = _load_reranker()
    pairs = [[query, doc.page_content] for doc in candidates]
    rerank_scores = reranker.predict(pairs)
    # 按得分降序取 topk
    docs_scored = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in docs_scored[:topk]]


def get_relevant_chunks(query: str, k: int = None):
    """
    返回 rerank 后最相关的文本 chunk 列表
    :param query: 用户查询
    :param k: top k 数量
    :return: List[str]
    """
    docs = get_topk_docs(query, k)
    return [doc.page_content for doc in docs]


def get_topk_docs_with_score(query: str, k: int = None):
    """
    返回带有 reranker 得分的 (Document, score) 列表

    :param query: 用户查询
    :param k: top k 数量
    :return: List[(Document, float)]
    """
    docs = get_topk_docs(query, k=k)
    reranker = _load_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    return list(zip(docs, scores))


'''def get_relevant_chunks(query: str, k: int = None):
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

    return patched'''

