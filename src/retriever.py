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