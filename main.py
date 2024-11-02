from pathlib import Path
import os
import pyalex
from pyalex import Works
import datetime
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Papers => embeddings => vector store
persist_directory = os.path.join(Path(__file__).resolve().parent, "db")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
if os.path.exists(persist_directory):
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    data_count = len(vector_db.get()["documents"])
    print(f"Vector store of {data_count} papers loaded.")
else:
    pyalex.config.email = "intoarmour@gmail.com"
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    year = datetime.datetime.now().year
    filter_params = {
        "topics.field.id": 17,
        "publication_year": f">{year-3}",
        "language": "en",
        "type": "article",
        "open_access.is_oa": True,
        "cited_by_count": ">20"}
    pager = Works().filter(**filter_params).paginate(per_page=200, n_max=None)
    i = 1
    for page in pager:
        for work in page:
            if work["title"] and work["abstract_inverted_index"]:
                sorted_words = sorted(work["abstract_inverted_index"].items(),
                                        key = lambda x: x[1][0])
                abstract = " ".join(word[0] for word in sorted_words)
                paper = "[Title]" + work["title"] + "[Abstract]" + abstract
                vector_db.add_documents([Document(page_content=paper)])
                print(f"Collecting data via PyAlex. {i} results completed.")
                i += 1
    print("Vector store saved.")
retriever = vector_db.as_retriever(search_kwargs={"k":5})
#

# Chat generation model
model_path = hf_hub_download(repo_id="taide/Llama3-TAIDE-LX-8B-Chat-Alpha1-4bit",
                             filename="taide-8b-a.3-q4_k_m.gguf")
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=8192,
    max_tokens=512,
    f16_kv=True,
    verbose=False,
    n_gpu_layers=-1,
    temperature=0.1,
    top_p=0.7,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),)
#

few_shot_prompt = """
你是一位論文解析專家，請為以下文章，列出原文標題(Title)，並根據摘要(Abstract)，用一句中文簡短說明研究主題及使用到的技術。
不要衍伸摘要中沒提到的內容，請僅基於可讀的部分進行摘要。 
=== 
輸入: 
[Title] Optimizing Distributed Systems with Reinforcement Learning 
[Abstract] This paper explores the use of reinforcement learning (RL) techniques to optimize the performance of distributed computing systems. By modeling system components as agents, the approach leverages RL algorithms to adaptively allocate resources and balance workloads across distributed nodes, resulting in improved system efficiency and reduced latency. 
輸出: 
- Optimizing Distributed Systems with Reinforcement Learning: 此論文探討強化學習在分散式系統中的作用，研究將系統元件建模為代理，使用強化學習算法自適應地分配資源並平衡負載。 
=== 
輸入: 
[Title] A Study on Blockchain Technology for Secure Data Sharing 
[Abstract] This paper investigates the application of blockchain technology to enhance data security in distributed systems. The proposed framework uses cryptographic techniques to ensure data integrity and privacy while allowing secure sharing among multiple stakeholders. The effectiveness of the framework is demonstrated through various case studies and performance evaluations. 
輸出: 
- A Study on Blockchain Technology for Secure Data Sharing: 此論文研究區塊鏈技術在分散式系統中增強資料安全性的應用，提出了一個使用加密技術以確保資料完整性與隱私的框架。 
===
"""

while True:
    user_input = input(">>> ")
    # RAG
    print("開始檢索。")
    results = retriever.invoke(user_input)
    contexts = list()
    for result in results:
        contexts.append(result.page_content)
    contexts = list(set(contexts))
    print("已檢索到相關文章，正在生成摘要...")
    for context in contexts:
        #Combine prompts
        context = context.replace("[Title]", "[Title] ").replace("[Abstract]", "\n[Abstract] ")
        prompt = few_shot_prompt + "輸入: \n" + context + " \n輸出: \n"
        #
        llm.invoke(prompt)
        print("\n")
    #