import os
from dotenv import load_dotenv

load_dotenv()  # 自动从 .env 文件加载变量

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
'''EMBEDDING_MODEL = "text-embedding-3-small"'''
EMBEDDING_MODEL = "fine_tuned_mpnet_base_v2" 
CHAT_MODEL = "gpt-4o-mini" #"gpt-3.5-turbo"
KNOWLEDGE_FOLDER = "./documents"
INDEX_PATH = "embeddings/defaultIndex"

CHUNK_SIZE=200
CHUNK_OVERLAP=50
TOP_K=5
