import os
from dotenv import load_dotenv

load_dotenv()  # 自动从 .env 文件加载变量

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
KNOWLEDGE_FOLDER = "./documents"
INDEX_PATH = "embeddings/defaultIndex"

CHUNK_SIZE=200
CHUNK_OVERLAP=50
TOP_K=5
