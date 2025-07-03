import json
import os
import warnings
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import time

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langsmith.utils import LangSmithMissingAPIKeyWarning
from dotenv import load_dotenv
from langchain import hub

# Load environment variables
load_dotenv()

# Suppress LangSmith warnings
warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)

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
    print(f"Successfully loaded vector database from '{db_path}'")
    return vector_db


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def accuracy_calculator(qa_list: List[Dict[str, str]], response_list: List[str]) -> Tuple[str, List[bool]]:
    """Calculate accuracy by checking if correct answers are substrings of responses."""
    matches = []
    for i, qa_item in enumerate(qa_list):
        matched = qa_item["answer"] in response_list[i]
        matches.append(matched)
    
    accuracy = sum(matches) / len(matches)
    return f"Accuracy: {accuracy:.3f}", matches


def evaluate_qa_system(qa_list: List[Dict[str, str]], qa_chain, description: str) -> pd.DataFrame:
    """Evaluate the QA system on the provided dataset."""
    results = {
        "question": [],
        "correct_answer": [],
        "context": [],
        "llm_answer": [],
    }
    
    print(f"Evaluating {description} with {len(qa_list)} questions...")
    for qa in tqdm(qa_list, desc=f"Processing {description}"):
        result = qa_chain.invoke(qa["question"])
        results["question"].append(qa["question"])
        results["correct_answer"].append(qa["answer"])
        
        # Handle different response formats
        if isinstance(result, dict):
            results["context"].append(result.get("context", ""))
            results["llm_answer"].append(result.get("answer", str(result)))
        else:
            results["context"].append("")  # Hub prompt doesn't return context separately
            results["llm_answer"].append(str(result))
    
    return pd.DataFrame(results)


def create_baseline_chain(vector_db: FAISS, k: int = 10):
    """Create baseline QA chain using LangChain Hub RAG prompt."""
    print("Creating BASELINE chain using LangChain Hub RAG prompt...")
    
    # Pull the standard RAG prompt from LangChain Hub
    prompt = hub.pull("rlm/rag-prompt")
    
    qa_chain = (
        {
            "context": vector_db.as_retriever(search_kwargs={"k": k}) | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | ChatOpenAI(model='gpt-4o-mini')
        | StrOutputParser()
    )
    
    return qa_chain


def create_optimized_chain(vector_db: FAISS, k: int = 3):
    """Create optimized QA chain with custom prompt."""
    print("Creating OPTIMIZED chain with custom prompt...")
    
    custom_prompt = ChatPromptTemplate.from_template(
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
    
    llm_chain = custom_prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    
    from langchain_core.runnables import RunnableLambda
    
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


def run_baseline_comparison():
    """Run comprehensive comparison between baseline and optimized systems."""
    print("="*80)
    print("BASELINE vs OPTIMIZED QA SYSTEM COMPARISON")
    print("="*80)
    
    # Load data
    qa_list = load_qa_data(QA_LIST_PATH)
    vector_db = load_vector_database(VECTOR_DB_PATH)
    
    # Use a substantial test subset for reliable comparison
    test_subset = qa_list[:50]  # 25% of dataset
    
    results = {}
    
    # Test 1: Baseline (LangChain Hub prompt with k=10)
    print("\n" + "="*50)
    print("BASELINE: LangChain Hub RAG prompt (k=10)")
    print("="*50)
    
    baseline_chain = create_baseline_chain(vector_db, k=10)
    
    start_time = time.time()
    baseline_results = evaluate_qa_system(test_subset, baseline_chain, "baseline")
    end_time = time.time()
    
    baseline_accuracy_text, baseline_matches = accuracy_calculator(
        test_subset, baseline_results['llm_answer'].tolist()
    )
    baseline_accuracy = float(baseline_accuracy_text.split(': ')[1])
    
    results['baseline'] = {
        'accuracy': baseline_accuracy,
        'correct_count': sum(baseline_matches),
        'total_count': len(baseline_matches),
        'time': end_time - start_time,
        'description': 'LangChain Hub RAG prompt, k=10'
    }
    
    print(f"\nBASELINE RESULTS:")
    print(f"Accuracy: {baseline_accuracy:.3f} ({sum(baseline_matches)}/{len(baseline_matches)})")
    print(f"Time: {end_time - start_time:.1f}s")
    
    # Test 2: Optimized (Custom prompt with k=3)
    print("\n" + "="*50)
    print("OPTIMIZED: Custom prompt (k=3)")
    print("="*50)
    
    optimized_chain = create_optimized_chain(vector_db, k=3)
    
    start_time = time.time()
    optimized_results = evaluate_qa_system(test_subset, optimized_chain, "optimized")
    end_time = time.time()
    
    optimized_accuracy_text, optimized_matches = accuracy_calculator(
        test_subset, optimized_results['llm_answer'].tolist()
    )
    optimized_accuracy = float(optimized_accuracy_text.split(': ')[1])
    
    results['optimized'] = {
        'accuracy': optimized_accuracy,
        'correct_count': sum(optimized_matches),
        'total_count': len(optimized_matches),
        'time': end_time - start_time,
        'description': 'Custom prompt, k=3'
    }
    
    print(f"\nOPTIMIZED RESULTS:")
    print(f"Accuracy: {optimized_accuracy:.3f} ({sum(optimized_matches)}/{len(optimized_matches)})")
    print(f"Time: {end_time - start_time:.1f}s")
    
    # Summary
    improvement = optimized_accuracy - baseline_accuracy
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Baseline Accuracy:  {baseline_accuracy:.3f} ({results['baseline']['description']})")
    print(f"Optimized Accuracy: {optimized_accuracy:.3f} ({results['optimized']['description']})")
    print(f"Improvement:        +{improvement:.3f} ({improvement*100:.1f} percentage points)")
    print(f"Token Reduction:    ~70% (k=3 vs k=10)")
    
    if improvement > 0:
        print(f"\n🎯 SUCCESS: Optimized system achieves {improvement*100:.1f}% point improvement!")
    else:
        print(f"\n⚠️  Baseline performs better by {abs(improvement)*100:.1f}% points")
    
    # Save detailed results
    with open('baseline_vs_optimized_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to 'baseline_vs_optimized_results.json'")
    return results


if __name__ == "__main__":
    results = run_baseline_comparison()