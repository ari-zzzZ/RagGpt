import sys
from src.ingest import build_vector_store
from src.chat import generate_answer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <你的问题>")
        sys.exit(1)

    query = sys.argv[1]

    # 若第一次运行，需要构建向量库（否则注释掉）
    #build_vector_store()

    answer = generate_answer(query)
    print("\n 回答：", answer)
