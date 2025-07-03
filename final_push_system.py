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
    print(f"Successfully loaded vector database from '{db_path}'")
    return vector_db


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_final_prompt() -> ChatPromptTemplate:
    """
    Final optimized prompt - back to what worked but with slight improvements.
    """
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
    """Create the final optimized QA chain - minimal changes to what worked."""
    print(f"Creating FINAL OPTIMIZED chain with k={k}")
    
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


def evaluate_final_system(qa_list: List[Dict[str, str]], qa_chain, description: str) -> pd.DataFrame:
    """Evaluate the final system."""
    results = {
        "question": [],
        "correct_answer": [],
        "context": [],
        "llm_answer": []
    }
    
    print(f"Evaluating {description} with {len(qa_list)} questions...")
    for qa in tqdm(qa_list, desc=f"Processing {description}"):
        result = qa_chain.invoke(qa["question"])
        
        results["question"].append(qa["question"])
        results["correct_answer"].append(qa["answer"])
        results["context"].append(result["context"])
        results["llm_answer"].append(result["answer"])
    
    return pd.DataFrame(results)


def test_different_k_values():
    """Test k=1, k=2, k=3 to find the absolute best."""
    print("🎯 TESTING DIFFERENT K VALUES FOR MAXIMUM ACCURACY")
    print("="*60)
    
    qa_list = load_qa_data(QA_LIST_PATH)
    vector_db = load_vector_database(VECTOR_DB_PATH)
    
    # Test on larger subset for reliable results
    test_subset = qa_list[:50]
    
    results = {}
    
    for k in [1, 2, 3, 4]:
        print(f"\n--- Testing k={k} ---")
        qa_chain = create_final_qa_chain(vector_db, k=k)
        
        start_time = time.time()
        eval_results = evaluate_final_system(test_subset, qa_chain, f"k={k}")
        end_time = time.time()
        
        accuracy_text, matches = accuracy_calculator(test_subset, eval_results['llm_answer'].tolist())
        accuracy = float(accuracy_text.split(': ')[1])
        
        results[k] = {
            'accuracy': accuracy,
            'correct': sum(matches),
            'total': len(matches),
            'time': end_time - start_time
        }
        
        print(f"k={k}: {accuracy:.3f} accuracy ({sum(matches)}/{len(matches)}), {end_time-start_time:.1f}s")
    
    # Find best k
    best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_k]['accuracy']
    
    print(f"\n🏆 BEST RESULT: k={best_k} with {best_accuracy:.3f} accuracy")
    
    return best_k, best_accuracy, results


def run_full_test_with_best_k(best_k: int):
    """Run full test with the best k value."""
    print(f"\n🚀 RUNNING FULL TEST WITH OPTIMAL k={best_k}")
    print("="*60)
    
    qa_list = load_qa_data(QA_LIST_PATH)
    vector_db = load_vector_database(VECTOR_DB_PATH)
    
    # Test on larger subset
    test_subset = qa_list[:100]  # 50% of dataset
    
    qa_chain = create_final_qa_chain(vector_db, k=best_k)
    
    start_time = time.time()
    results = evaluate_final_system(test_subset, qa_chain, f"FINAL k={best_k}")
    end_time = time.time()
    
    accuracy_text, matches = accuracy_calculator(test_subset, results['llm_answer'].tolist())
    accuracy = float(accuracy_text.split(': ')[1])
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Accuracy: {accuracy:.3f} ({sum(matches)}/{len(matches)})")
    print(f"Time: {end_time - start_time:.1f}s")
    
    # Compare to all previous results
    baseline = 0.43
    previous_best = 0.77
    
    print(f"\n📊 COMPREHENSIVE COMPARISON:")
    print(f"Baseline (LangChain Hub): {baseline:.3f}")
    print(f"Previous best (k=3): {previous_best:.3f}")
    print(f"Final optimized (k={best_k}): {accuracy:.3f}")
    print(f"Improvement over baseline: +{accuracy-baseline:.3f} ({(accuracy-baseline)*100:.1f} points)")
    print(f"Improvement over previous: +{accuracy-previous_best:.3f} ({(accuracy-previous_best)*100:.1f} points)")
    
    if accuracy >= 0.85:
        print(f"🎉 MISSION ACCOMPLISHED! {accuracy:.3f} ≥ 85% target!")
    elif accuracy >= 0.80:
        print(f"🎯 EXCELLENT! {accuracy:.3f} ≥ 80% original target!")
    elif accuracy > previous_best:
        print(f"✅ PROGRESS! {accuracy:.3f} > {previous_best:.3f} previous best!")
    
    return accuracy, results


def main():
    """Main execution with systematic testing."""
    print("🚀" * 30)
    print("FINAL PUSH FOR MAXIMUM ACCURACY!")
    print("🚀" * 30)
    
    # Step 1: Find optimal k
    best_k, best_accuracy, k_results = test_different_k_values()
    
    # Step 2: Run full test with best k
    final_accuracy, final_results = run_full_test_with_best_k(best_k)
    
    # Step 3: Save results
    summary = {
        'k_value_testing': k_results,
        'best_k': best_k,
        'final_accuracy': final_accuracy,
        'target_achieved': final_accuracy >= 0.80,
        'stretch_target_achieved': final_accuracy >= 0.85
    }
    
    with open('final_push_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to 'final_push_results.json'")
    
    return final_accuracy


if __name__ == "__main__":
    final_accuracy = main()