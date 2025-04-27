from openai import OpenAI
from config import settings
from src.retriever import get_topk_docs

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_answer(
    query: str,
    model: str = None,
    top_k: int = None
) -> str:
    """
    基于 RAG 执行检索并调用 OpenAI 接口生成回答。
    可通过参数覆盖模型和检索条数，同时打印检索到的文档片段。
    """
    docs = get_topk_docs(query, k=top_k)
    print("检索到的文档片段：")
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        print(f"[{idx}] 来源：{source}\n{doc.page_content}\n{'-'*40}")
    chunks = [doc.page_content for doc in docs]
    context = "\n\n".join(chunks)

    messages = [
        {"role": "system", "content": "你是一个专业问答助手。"},
        {"role": "user", "content": f"以下是与用户问题相关的资料：\n{context}"},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model=model or settings.CHAT_MODEL,
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content