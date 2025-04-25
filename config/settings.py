import os
from dotenv import load_dotenv

load_dotenv()  # 自动从 .env 文件加载变量

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
KNOWLEDGE_FOLDER = "./documents"
INDEX_PATH = "embeddings/index.faiss"
