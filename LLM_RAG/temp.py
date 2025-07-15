from langchain_community.chat_models import QianfanChatEndpoint
from ragas.llms import LangchainLLM
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

chat = QianfanChatEndpoint(model=model, qianfan_ak=QIANFAN_AK, qianfan_sk=QIANFAN_SK, **model_kwargs)
v_llm = LangchainLLM(chat)

v_embeddings = QianfanEmbeddingsEndpoint(
               qianfan_ak=QIANFAN_AK,
               qianfan_sk=QIANFAN_SK,
           )

# 然后重新指定各评价指标使用的llm
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

faithfulness.llm = v_llm
answer_relevancy.llm = v_llm
answer_relevancy.embeddings = v_embeddings

context_recall.llm = v_llm
context_precision.llm = v_llm

# 重新一键式测评
result = evaluate(
    evalsets,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

df = result.to_pandas()
df.head()