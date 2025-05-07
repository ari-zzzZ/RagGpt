'''import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from config import settings
from pathlib import Path
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# 延迟加载
_db = None
#_embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
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

    # 阶段1: BM25 初筛（关键词召回）
    bm25_k = int(topk * getattr(settings, 'BM25_INIT_MULTIPLIER', 12))  # 缩小默认倍数
    bm25, docs = _load_bm25()
    tokenized = query.split()
    scores = bm25.get_scores(tokenized)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bm25_k]
    bm25_docs = [docs[i] for i in idxs]

    # 阶段2: Dense 向量检索（语义召回）
    dense_k = int(topk * getattr(settings, 'DENSE_INIT_MULTIPLIER', 12))  # 增加倍数覆盖更多
    dense_docs = db.similarity_search(query, k=dense_k)

    # 交集优先：选取同时命中 BM25 和 dense 的文档
    bm25_ids = {doc.metadata.get('source') or doc.metadata.get('id') for doc in bm25_docs}
    intersec = [doc for doc in dense_docs if (doc.metadata.get('source') or doc.metadata.get('id')) in bm25_ids]
    if len(intersec) >= topk:
        candidates = intersec
    else:
        # 否则合并 BM25 & dense 去重
        unique = {}
        for doc in bm25_docs + dense_docs:
            key = doc.metadata.get('source') or doc.metadata.get('id')
            unique[key] = doc
        candidates = list(unique.values())

    if not candidates:
        return []

    # 阶段3: Cross-Encoder 重排序
    reranker = _load_reranker()
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    docs_scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    # 返回 top k 文档
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
    return list(zip(docs, scores))'''

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from config import settings

# 延迟加载全局单例
_db = None
_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
_reranker = None
_bm25 = None
_bm25_docs = None


def _load_db():
    """懒加载 FAISS 向量库"""
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
    """懒加载 BM25 索引，用于关键词初筛"""
    global _bm25, _bm25_docs
    if _bm25 is None:
        db = _load_db()
        _bm25_docs = list(db.docstore._dict.values())
        texts = [doc.page_content.split() for doc in _bm25_docs]
        _bm25 = BM25Okapi(texts)
    return _bm25, _bm25_docs


def _load_reranker():
    """懒加载 Cross-Encoder 重排序模型"""
    global _reranker
    if _reranker is None:
        model_name = getattr(settings, 'RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        _reranker = CrossEncoder(model_name)
    return _reranker


def get_topk_docs(query: str, k: int = None):
    """混合检索：BM25 + Dense 融合后，Cross-Encoder 精排，优先交集并补满"""
    topk = k or settings.TOP_K
    db = _load_db()

    # 1. BM25 初筛
    bm25_k = int(topk * getattr(settings, 'BM25_INIT_MULTIPLIER', 12))
    bm25, docs = _load_bm25()
    tokenized = query.split()
    bm25_scores = bm25.get_scores(tokenized)
    bm25_idxs = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_k]
    bm25_docs = [docs[i] for i in bm25_idxs]

    # 2. Dense 检索
    dense_k = int(topk * getattr(settings, 'DENSE_INIT_MULTIPLIER', 12))
    dense_docs = db.similarity_search(query, k=dense_k)

    # 交集逻辑
    bm25_ids = {doc.metadata.get('source') or doc.metadata.get('id') for doc in bm25_docs}
    intersec = [doc for doc in dense_docs if (doc.metadata.get('source') or doc.metadata.get('id')) in bm25_ids]
    print(f"[Retriever] Intersection count: {len(intersec)}")

    # 补齐剩余
    unique = {}
    for doc in bm25_docs + dense_docs:
        key = doc.metadata.get('source') or doc.metadata.get('id')
        unique[key] = doc
    rest = [doc for key, doc in unique.items() if key not in bm25_ids]

    # 最终候选：交集 + rest
    candidates = intersec + rest

    if not candidates:
        return []

    # 3. Cross-Encoder 精排
    rerank_k = int(topk * getattr(settings, 'RERANK_INIT_MULTIPLIER', 8))
    to_rerank = candidates[:rerank_k]
    reranker = _load_reranker()
    pairs = [[query, doc.page_content] for doc in to_rerank]
    ce_scores = reranker.predict(pairs)
    docs_scored = sorted(zip(to_rerank, ce_scores), key=lambda x: x[1], reverse=True)

    # 返回 topk
    return [doc for doc, _ in docs_scored[:topk]]


def get_relevant_chunks(query: str, k: int = None):
    docs = get_topk_docs(query, k)
    return [doc.page_content for doc in docs]


def get_topk_docs_with_score(query: str, k: int = None):
    docs = get_topk_docs(query, k)
    reranker = _load_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    return list(zip(docs, scores))
