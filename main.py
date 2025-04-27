import argparse
from src.ingest import build_vector_store
from src.chat import generate_answer

def main():
    parser = argparse.ArgumentParser(description="RAG 问答系统 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-index 子命令
    parser_build = subparsers.add_parser("build-index", help="构建或增量更新向量索引")

    # ask 子命令
    parser_ask = subparsers.add_parser("ask", help="提问并获取回答")
    parser_ask.add_argument("query", help="用户的提问内容")
    parser_ask.add_argument(
        "--topk", type=int, default=None,
        help="检索的 chunk 数量，覆盖默认值"
    )
    parser_ask.add_argument(
        "--model", type=str, default=None,
        help="OpenAI 使用的模型，覆盖默认值"
    )

    args = parser.parse_args()
    if args.command == "build-index":
        build_vector_store()
    elif args.command == "ask":
        answer = generate_answer(
            args.query,
            model=args.model,
            top_k=args.topk
        )
        print("\n回答：", answer)

if __name__ == "__main__":
    main()
