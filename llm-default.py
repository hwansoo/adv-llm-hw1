import os
import json
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import warnings
from langsmith.utils import LangSmithMissingAPIKeyWarning
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai.chat_models import ChatOpenAI
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)

with open("qa_list.json", "r") as f:
    qa_list = json.load(f)

save_folder = "faiss_index"
embeddings = OpenAIEmbeddings()

# allow_dangerous_deserialization=True는 LangChain v0.2.0 이상에서 pickle 파일 로드 시 필요
loaded_vector_db = FAISS.load_local(
    save_folder, embeddings, allow_dangerous_deserialization=True
)
print(f"'{save_folder}'에서 VectorDB를 성공적으로 불러왔습니다.")


def accuracy_calculator(qa_list, response_list):
    cnt_true = 0
    cnt_false = 0
    matches = []
    for i, qa_i in enumerate(qa_list):
        matched = qa_i["answer"] in response_list[i]
        if matched:
            cnt_true += 1
            matches.append(True)
        else:
            cnt_false += 1
            matches.append(False)
    return f"Accuracy : {cnt_true/(cnt_true+cnt_false):.2f}", matches


prompt = ChatPromptTemplate.from_template(
    """
You are the top expert in question-answering tasks. Provide an accurate and useful answer to the question using the retrieved context.

Context:
{context}

Instructions:
* The answer must be an exact substring of the given context.
* DO NOT rephrase, even if there are errors in the context.
* DO NOT omit any punctuation, including, but not limited to, full stops, commas, and quotation marks.
* If the answer is not found in the context, respond with "No Answer".
* The answer must be written as one or up to three full sentences.

Question: {question}

Answer:
"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = loaded_vector_db.as_retriever(search_kwargs={"k": 10})

llm_chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

qa_chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough(),
} | RunnableLambda(
    lambda inputs: {
        "question": inputs["question"],
        "context": inputs["context"],
        "answer": llm_chain.invoke(
            {"context": inputs["context"], "question": inputs["question"]}
        ),
    }
)

questions = []
correct_answers = []
contexts = []
llm_answers = []

for qa in tqdm(qa_list):
    result = qa_chain.invoke(qa["question"])
    questions.append(qa["question"])
    correct_answers.append(qa["answer"])
    contexts.append(result["context"])
    llm_answers.append(result["answer"])


print("")
accuracy_text, match_tf = accuracy_calculator(qa_list, llm_answers)
print(accuracy_text)

results_df = pd.DataFrame(
    {
        "question": questions,
        "correct_answer": correct_answers,
        "context": contexts,
        "llm_answer": llm_answers,
        "is_correct": match_tf,
    }
)
