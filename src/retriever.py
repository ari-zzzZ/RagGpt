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
    bm25_k = int(topk * getattr(settings, 'BM25_INIT_MULTIPLIER', 8))
    bm25, docs = _load_bm25()
    tokenized = query.split()
    bm25_scores = bm25.get_scores(tokenized)
    bm25_idxs = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_k]
    bm25_docs = [docs[i] for i in bm25_idxs]

    # 2. Dense 检索
    dense_k = int(topk * getattr(settings, 'DENSE_INIT_MULTIPLIER', 8))
    dense_docs = db.similarity_search(query, k=dense_k)

    # 交集逻辑
    bm25_ids = {doc.metadata.get('source') or doc.metadata.get('id') for doc in bm25_docs}
    intersec = [doc for doc in dense_docs if (doc.metadata.get('source') or doc.metadata.get('id')) in bm25_ids]
    print(f"[Retriever] Intersection (交集) count: {len(intersec)}")

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
    '''
    总结优先级:
    第一优先：交集（BM25 与 Dense 都命中的文档）
    第二优先：剩余的 BM25-only 文档，按 BM25 原始排序
    第三优先：剩余的 Dense-only 文档，按 Dense 原始排序.

    这样就保证了：既不会把语义召回但关键词召回不到的“孤儿”文档提前，
    也不会让只有关键词命中的文档压过更语义相关的候选。
    '''

    # 3. Cross-Encoder 精排
    rerank_k = int(topk * getattr(settings, 'RERANK_INIT_MULTIPLIER', 10))
    to_rerank = candidates[:rerank_k]
    reranker = _load_reranker()
    pairs = [[query, doc.page_content] for doc in to_rerank]
    ce_scores = reranker.predict(pairs)
    docs_scored = sorted(zip(to_rerank, ce_scores), key=lambda x: x[1], reverse=True)

    # 返回 topk
    #return [doc for doc, _ in docs_scored[:topk]]
    return db.similarity_search(query, k=topk)


def get_relevant_chunks(query: str, k: int = None):
    docs = get_topk_docs(query, k)
    return [doc.page_content for doc in docs]


def get_topk_docs_with_score(query: str, k: int = None):
    docs = get_topk_docs(query, k)
    reranker = _load_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    return list(zip(docs, scores))
