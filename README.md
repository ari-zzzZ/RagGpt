1.请在根目录新建.env文件，设置你的openai api key

2.将知识库文档放入documents/中

3.然后执行：”python main.py build-index”， 
这会对新增文件做分块、Embedding 并追加到已有 FAISS 索引，并显示文件数、chunk数。

然后使用以下格式提问：“python main.py ask "中国2024年经济增速是多少？”

你可以指定召回的文档数以及模型，例如：
python main.py ask "..." --topk 6
python main.py ask "..." --model gpt-4

使用愉快！

