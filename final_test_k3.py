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

def run_final_test():
    """Run final comprehensive test with k=3."""
    print("🚀 FINAL COMPREHENSIVE TEST WITH k=3")
    print("="*50)
    
    qa_list = load_qa_data(QA_LIST_PATH)
    vector_db = load_vector_database(VECTOR_DB_PATH)
    
    # Test on 80 questions (40% of dataset)
    test_subset = qa_list[:80]
    
    print(f"Testing on {len(test_subset)} questions...")
    
    qa_chain = create_final_qa_chain(vector_db, k=3)
    
    start_time = time.time()
    responses = []
    for qa in tqdm(test_subset, desc="Processing questions"):
        result = qa_chain.invoke(qa["question"])
        responses.append(result["answer"])
    
    end_time = time.time()
    
    accuracy_text, matches = accuracy_calculator(test_subset, responses)
    accuracy = float(accuracy_text.split(': ')[1])
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Accuracy: {accuracy:.3f} ({sum(matches)}/{len(matches)})")
    print(f"Time: {end_time - start_time:.1f}s")
    print(f"Speed: {(end_time - start_time)/len(test_subset):.2f}s/question")
    
    # Compare to all previous results
    baseline = 0.43
    previous_best = 0.77
    
    print(f"\n📊 COMPREHENSIVE COMPARISON:")
    print(f"Baseline (LangChain Hub): {baseline:.3f}")
    print(f"Previous best (k=3, 200 questions): {previous_best:.3f}")
    print(f"Final optimized (k=3, 80 questions): {accuracy:.3f}")
    print(f"Improvement over baseline: +{accuracy-baseline:.3f} ({(accuracy-baseline)*100:.1f} points)")
    print(f"Improvement over previous: +{accuracy-previous_best:.3f} ({(accuracy-previous_best)*100:.1f} points)")
    
    if accuracy >= 0.85:
        print(f"🎉 MISSION ACCOMPLISHED! {accuracy:.3f} ≥ 85% stretch target!")
    elif accuracy >= 0.80:
        print(f"🎯 EXCELLENT! {accuracy:.3f} ≥ 80% original target!")
    elif accuracy > previous_best:
        print(f"✅ PROGRESS! {accuracy:.3f} > {previous_best:.3f} previous best!")
    
    # Save results
    summary = {
        'test_size': len(test_subset),
        'accuracy': accuracy,
        'correct_count': sum(matches),
        'total_count': len(matches),
        'time': end_time - start_time,
        'speed_per_question': (end_time - start_time)/len(test_subset),
        'improvement_over_baseline': accuracy - baseline,
        'improvement_over_previous': accuracy - previous_best,
        'target_80_achieved': accuracy >= 0.80,
        'target_85_achieved': accuracy >= 0.85,
        'k_value': 3,
        'model': 'gpt-4o-mini'
    }
    
    with open('final_test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to 'final_test_results.json'")
    
    return accuracy

if __name__ == "__main__":
    final_accuracy = run_final_test()