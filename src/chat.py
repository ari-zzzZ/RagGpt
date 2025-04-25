from openai import OpenAI
from config import settings
from src.retriever import get_relevant_chunks

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_answer(query):
    try:
        chunks = get_relevant_chunks(query)
        print(" 检索到的 chunks：", chunks)
        context = "\n\n".join(chunks)
        prompt = f"""
以下是与用户问题相关的资料：
{context}

请基于上述内容回答用户的问题：
"""
        response = client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业问答助手。"},
                {"role": "user", "content": prompt + query}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(" OpenAI 请求失败：", e)
        return "无法生成回答。"