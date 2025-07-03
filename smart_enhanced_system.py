import json
import os
import warnings
import re
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
from langsmith.utils import LangSmithMissingAPIKeyWarning
from dotenv import load_dotenv

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


def smart_clean_answer(raw_answer: str) -> str:
    """
    Smart but conservative answer cleaning.
    Only remove obvious prefixes while preserving the core answer.
    """
    if not raw_answer or not raw_answer.strip():
        return "No Answer"
    
    answer = raw_answer.strip()
    
    # Handle "No Answer" cases
    if "no answer" in answer.lower():
        return "No Answer"
    
    # Only remove very obvious prefixes - be conservative!
    obvious_prefixes = [
        "according to the context, ",
        "based on the context, ",
        "the answer is ",
        "the context states that ",
        "the passage says ",
        "the text mentions "
    ]
    
    answer_lower = answer.lower()
    for prefix in obvious_prefixes:
        if answer_lower.startswith(prefix):
            answer = answer[len(prefix):].strip()
            break  # Only remove one prefix
    
    # Remove wrapping quotes only if they wrap the ENTIRE answer
    if (answer.startswith('"') and answer.endswith('"') and answer.count('"') == 2) or \
       (answer.startswith("'") and answer.endswith("'") and answer.count("'") == 2):
        answer = answer[1:-1].strip()
    
    return answer


def create_smart_prompt() -> ChatPromptTemplate:
    """
    Create a smarter prompt that focuses on exact extraction with examples.
    Less aggressive than before, more focused on precision.
    """
    return ChatPromptTemplate.from_template(
        """
You are an expert at finding exact answers in text. Extract the precise answer that directly answers the question.

Context:
{context}

IMPORTANT RULES:
1. Your answer must be an EXACT substring from the context above
2. Copy the text exactly as it appears - same punctuation, capitalization, spacing
3. Do NOT add explanations or extra words
4. If the answer is not in the context, respond "No Answer"

GOOD EXAMPLES:
Question: What year did something happen?
Context: "...it happened in 1905..."
Answer: 1905

Question: Who invented the device?
Context: "...Alexander Graham Bell invented the telephone..."
Answer: Alexander Graham Bell

Question: What is the missing information?
Context: "...discusses many topics but not this specific one..."
Answer: No Answer

Now extract the exact answer:

Question: {question}

Answer:
"""
    )


def create_smart_qa_chain(vector_db: FAISS, k: int = 3):
    """Create smart QA chain with focused improvements."""
    print(f"Creating SMART ENHANCED chain with k={k}")
    
    prompt = create_smart_prompt()
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
                "raw_answer": llm_chain.invoke(
                    {"context": inputs["context"], "question": inputs["question"]}
                ),
            }
        )
        | RunnableLambda(
            lambda inputs: {
                "question": inputs["question"],
                "context": inputs["context"],
                "raw_answer": inputs["raw_answer"],
                "answer": smart_clean_answer(inputs["raw_answer"]),
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


def evaluate_smart_system(qa_list: List[Dict[str, str]], qa_chain, description: str) -> pd.DataFrame:
    """Evaluate the smart enhanced QA system."""
    results = {
        "question": [],
        "correct_answer": [],
        "context": [],
        "raw_answer": [],
        "final_answer": []
    }
    
    print(f"Evaluating {description} with {len(qa_list)} questions...")
    for qa in tqdm(qa_list, desc=f"Processing {description}"):
        result = qa_chain.invoke(qa["question"])
        
        results["question"].append(qa["question"])
        results["correct_answer"].append(qa["answer"])
        results["context"].append(result["context"])
        results["raw_answer"].append(result["raw_answer"])
        results["final_answer"].append(result["answer"])
    
    return pd.DataFrame(results)


def run_smart_evaluation():
    """Run smart enhanced evaluation with conservative improvements."""
    print("🎯" * 40)
    print("SMART ENHANCED QA SYSTEM - CONSERVATIVE BUT EFFECTIVE!")
    print("🎯" * 40)
    
    # Load data
    qa_list = load_qa_data(QA_LIST_PATH)
    vector_db = load_vector_database(VECTOR_DB_PATH)
    
    # Test on subset first
    test_subset = qa_list[:100]  # 50% of dataset
    
    print(f"\nTesting smart enhanced system on {len(test_subset)} questions...")
    print("Smart improvements:")
    print("✅ Improved prompt with clear extraction rules")
    print("✅ Conservative answer cleaning (only obvious prefixes)")
    print("✅ Focus on exact substring matching")
    print("✅ No complex ensemble - just optimized k=3")
    print("✅ Better 'No Answer' detection")
    
    # Create smart enhanced system
    smart_chain = create_smart_qa_chain(vector_db, k=3)
    
    # Evaluate
    start_time = time.time()
    results = evaluate_smart_system(test_subset, smart_chain, "SMART ENHANCED")
    end_time = time.time()
    
    # Calculate accuracy
    accuracy_text, matches = accuracy_calculator(test_subset, results['final_answer'].tolist())
    accuracy = float(accuracy_text.split(': ')[1])
    
    # Results
    print(f"\n🎯 SMART ENHANCED RESULTS:")
    print(f"Accuracy: {accuracy:.3f} ({sum(matches)}/{len(matches)})")
    print(f"Time: {end_time - start_time:.1f}s ({(end_time - start_time)/len(test_subset):.2f}s/question)")
    
    # Compare to previous results
    baseline_accuracy = 0.43  # From full dataset baseline
    optimized_accuracy = 0.77  # From our previous best
    
    print(f"\n📈 COMPARISON:")
    print(f"Baseline (LangChain Hub): {baseline_accuracy:.3f}")
    print(f"Previous optimized: {optimized_accuracy:.3f}")
    print(f"Smart enhanced: {accuracy:.3f}")
    print(f"Improvement over baseline: +{accuracy - baseline_accuracy:.3f}")
    print(f"Improvement over previous: +{accuracy - optimized_accuracy:.3f}")
    
    # Target analysis
    if accuracy >= 0.85:
        print(f"🎉 TARGET ACHIEVED! {accuracy:.3f} exceeds 85% target!")
    elif accuracy >= 0.80:
        print(f"🎯 GREAT! {accuracy:.3f} exceeds original 80% target!")
    elif accuracy > optimized_accuracy:
        print(f"✅ IMPROVEMENT! {accuracy:.3f} is better than previous best!")
    else:
        print(f"📊 Need more work. Current: {accuracy:.3f}, Target: 85%")
    
    # Detailed error analysis
    wrong_answers = []
    for i, (qa, match) in enumerate(zip(test_subset, matches)):
        if not match:
            raw_ans = results.iloc[i]['raw_answer']
            final_ans = results.iloc[i]['final_answer']
            context = results.iloc[i]['context']
            
            wrong_answers.append({
                'question': qa['question'],
                'expected': qa['answer'],
                'raw_answer': raw_ans,
                'final_answer': final_ans,
                'context_has_answer': qa['answer'] in context,
                'raw_has_answer': qa['answer'] in raw_ans,
                'cleaning_issue': qa['answer'] in raw_ans and qa['answer'] not in final_ans
            })
    
    if wrong_answers:
        print(f"\n🔍 DETAILED ERROR ANALYSIS ({len(wrong_answers)} errors):")
        
        context_issues = sum(1 for w in wrong_answers if not w['context_has_answer'])
        extraction_issues = sum(1 for w in wrong_answers if w['context_has_answer'] and not w['raw_has_answer'])
        cleaning_issues = sum(1 for w in wrong_answers if w['cleaning_issue'])
        
        print(f"Context issues (answer not in retrieved docs): {context_issues}")
        print(f"Extraction issues (answer in context but not extracted): {extraction_issues}")
        print(f"Cleaning issues (answer extracted but cleaned away): {cleaning_issues}")
        
        print(f"\nSample error analysis:")
        for i, error in enumerate(wrong_answers[:3]):
            print(f"\n{i+1}. Q: {error['question'][:60]}...")
            print(f"   Expected: '{error['expected']}'")
            print(f"   Raw LLM: '{error['raw_answer'][:80]}...'")
            print(f"   Final: '{error['final_answer']}'")
            print(f"   Issues: Context={'✅' if error['context_has_answer'] else '❌'}, "
                  f"Raw={'✅' if error['raw_has_answer'] else '❌'}, "
                  f"Cleaning={'❌' if error['cleaning_issue'] else '✅'}")
    
    # Save results
    results.to_csv('smart_enhanced_results.csv', index=False)
    
    summary = {
        'accuracy': accuracy,
        'correct_count': sum(matches),
        'total_count': len(matches),
        'time': end_time - start_time,
        'improvement_over_baseline': accuracy - baseline_accuracy,
        'improvement_over_previous': accuracy - optimized_accuracy,
        'target_85_achieved': accuracy >= 0.85,
        'target_80_achieved': accuracy >= 0.80,
        'error_breakdown': {
            'context_issues': context_issues if wrong_answers else 0,
            'extraction_issues': extraction_issues if wrong_answers else 0,
            'cleaning_issues': cleaning_issues if wrong_answers else 0
        }
    }
    
    with open('smart_enhanced_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   • smart_enhanced_results.csv")
    print(f"   • smart_enhanced_summary.json")
    
    return accuracy, results


if __name__ == "__main__":
    accuracy, results = run_smart_evaluation()