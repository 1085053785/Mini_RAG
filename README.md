# Mini_RAG
从零开始搭建的minirag，用来清楚原理
项目目标：本地文档问答
技术路线：bge-small-zh-v1.5 做 embedding，Qwen2.5-0.5B-Instruct 做生成
核心流程：query → retrieve → context → prompt → answer
已发现问题：知识库混杂会导致跨主题误召回
下一步改进：分库、metadata filter、rerank、query routing