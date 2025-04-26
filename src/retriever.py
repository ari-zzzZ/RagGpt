from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from config import settings


def load_vector_store():
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    return FAISS.load_local(settings.INDEX_PATH, embeddings,allow_dangerous_deserialization=True)


def get_relevant_chunks(query, k=4):
    db = load_vector_store()
    docs = db.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

#新建函数 get_topk_docs(query, k) 来用于评估
def get_topk_docs(query, k=4, db=None):
    if db is None:
        db = load_vector_store()
    return db.similarity_search(query, k=k)#  返回完整 doc 对象（含 doc_id）