import json
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import time

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
QA_LIST_PATH = "qa_list.json"
VECTOR_DB_PATH = "faiss_index"

def load_qa_data(file_path: str) -> List[Dict[str, str]]:
    """Load question-answer pairs from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_vector_database(db_path: str) -> FAISS:
    """Load pre-built FAISS vector database."""
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.load_local(
        db_path, embeddings, allow_dangerous_deserialization=True
    )
    return vector_db

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_final_prompt() -> ChatPromptTemplate:
    """Final optimized prompt."""
    return ChatPromptTemplate.from_template(
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

def create_final_qa_chain(vector_db: FAISS, k: int = 3):
    """Create the final optimized QA chain."""
    prompt = create_final_prompt()
    llm_chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    
    qa_chain = (
        {
            "context": vector_db.as_retriever(search_kwargs={"k": k}) | format_docs,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(
            lambda inputs: {
                "question": inputs["question"],
                "context": inputs["context"],
                "answer": llm_chain.invoke(
                    {"context": inputs["context"], "question": inputs["question"]}
                ),
            }
        )
    )
    
    return qa_chain

def accuracy_calculator(qa_list: List[Dict[str, str]], response_list: List[str]) -> Tuple[str, List[bool]]:
    """Calculate accuracy by checking if correct answers are substrings of responses."""
    matches = []
    for i, qa_item in enumerate(qa_list):
        matched = qa_item["answer"] in response_list[i]
        matches.append(matched)
    
    accuracy = sum(matches) / len(matches)
    return f"Accuracy: {accuracy:.3f}", matches

def quick_test():
    """Quick test with different k values."""
    print("🎯 QUICK K-VALUE TEST")
    print("="*40)
    
    qa_list = load_qa_data(QA_LIST_PATH)
    vector_db = load_vector_database(VECTOR_DB_PATH)
    
    # Test on small subset for speed
    test_subset = qa_list[:20]
    
    results = {}
    
    for k in [1, 2, 3, 4]:
        print(f"\n--- Testing k={k} ---")
        qa_chain = create_final_qa_chain(vector_db, k=k)
        
        responses = []
        for qa in test_subset:
            result = qa_chain.invoke(qa["question"])
            responses.append(result["answer"])
        
        accuracy_text, matches = accuracy_calculator(test_subset, responses)
        accuracy = float(accuracy_text.split(': ')[1])
        
        results[k] = accuracy
        print(f"k={k}: {accuracy:.3f} accuracy ({sum(matches)}/{len(matches)})")
    
    # Find best k
    best_k = max(results.keys(), key=lambda k: results[k])
    print(f"\n🏆 BEST: k={best_k} with {results[best_k]:.3f} accuracy")
    
    return best_k, results[best_k]

if __name__ == "__main__":
    best_k, best_accuracy = quick_test()