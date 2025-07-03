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


def clean_answer(raw_answer: str, max_words: int = 12) -> str:
    """
    Post-process LLM answer to extract clean, precise responses.
    
    This is the #1 improvement - most errors are due to verbose/imprecise extraction.
    """
    if not raw_answer or not raw_answer.strip():
        return "No Answer"
    
    answer = raw_answer.strip()
    
    # Handle "No Answer" cases
    if "no answer" in answer.lower():
        return "No Answer"
    
    # Remove common prefixes that LLMs add
    prefixes_to_remove = [
        "according to the context,",
        "based on the context,",
        "the answer is",
        "the context states that",
        "from the context,",
        "as mentioned in the context,",
        "the passage indicates that",
        "the text shows that"
    ]
    
    answer_lower = answer.lower()
    for prefix in prefixes_to_remove:
        if answer_lower.startswith(prefix):
            answer = answer[len(prefix):].strip()
            answer_lower = answer.lower()
    
    # Remove quotes if they wrap the entire answer
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1].strip()
    
    # If answer is too long, try to extract the core part
    words = answer.split()
    if len(words) > max_words:
        # Try to find the first sentence
        sentences = re.split(r'[.!?]', answer)
        if sentences and len(sentences[0].split()) <= max_words:
            answer = sentences[0].strip()
        else:
            # Truncate to max_words
            answer = ' '.join(words[:max_words])
    
    # Clean up extra whitespace and punctuation
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    return answer


def create_enhanced_prompt() -> ChatPromptTemplate:
    """
    Create enhanced prompt with specific examples and extraction instructions.
    
    This addresses the main issue: LLMs extracting wrong parts of context.
    """
    return ChatPromptTemplate.from_template(
        """
You are an expert at extracting EXACT answers from text. Your job is to find the precise answer substring.

Context:
{context}

EXTRACTION EXAMPLES:
Question: What year did Darwin publish his theory?
Context: "...Darwin published his groundbreaking theory in 1859 and it changed..."
✅ CORRECT: 1859
❌ WRONG: "Darwin published his groundbreaking theory in 1859"
❌ WRONG: "in 1859"

Question: Who invented the telephone?
Context: "...Alexander Graham Bell invented the telephone in 1876..."
✅ CORRECT: Alexander Graham Bell
❌ WRONG: "Alexander Graham Bell invented the telephone"

Question: What is the capital of France?
Context: "...many cities but nothing about French capital..."
✅ CORRECT: No Answer
❌ WRONG: "The context doesn't mention"

CRITICAL RULES:
1. Extract ONLY the direct answer - no extra words
2. Must be exact substring from context (copy-paste exactly)
3. No prefixes like "According to..." or "The answer is..."
4. No explanations or elaborations
5. If not found in context: "No Answer"
6. Maximum 1-3 words for most answers, up to one sentence for longer answers

Question: {question}

Extract the exact answer:
"""
    )


def create_ensemble_qa_chain(vector_db: FAISS, k_values: List[int] = [2, 3, 4]):
    """
    Create ensemble QA chain using multiple k values for robust predictions.
    """
    print(f"Creating ENSEMBLE chain with k values: {k_values}")
    
    prompt = create_enhanced_prompt()
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    def ensemble_invoke(question: str) -> Dict:
        """Get predictions from multiple k values and ensemble them."""
        predictions = []
        contexts = []
        
        for k in k_values:
            retriever = vector_db.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(question)
            context = format_docs(docs)
            
            # Get raw answer
            raw_answer = (prompt | llm | StrOutputParser()).invoke({
                "context": context,
                "question": question
            })
            
            # Clean the answer
            clean_ans = clean_answer(raw_answer)
            
            predictions.append({
                "k": k,
                "raw_answer": raw_answer,
                "clean_answer": clean_ans,
                "context": context
            })
            contexts.append(context)
        
        # Ensemble logic: majority vote, prefer shorter answers
        clean_answers = [p["clean_answer"] for p in predictions]
        
        # Count votes for each unique answer
        answer_votes = {}
        for ans in clean_answers:
            answer_votes[ans] = answer_votes.get(ans, 0) + 1
        
        # Get the most voted answer
        if answer_votes:
            ensemble_answer = max(answer_votes.keys(), key=lambda x: (
                answer_votes[x],  # Higher vote count
                -len(x.split())   # Shorter answer as tiebreaker
            ))
        else:
            ensemble_answer = clean_answers[0] if clean_answers else "No Answer"
        
        # Return the prediction with best context (from k=3, our optimal)
        best_prediction = next(p for p in predictions if p["k"] == 3)
        
        return {
            "question": question,
            "context": best_prediction["context"],
            "answer": ensemble_answer,
            "ensemble_details": predictions,
            "vote_counts": answer_votes
        }
    
    return ensemble_invoke


def accuracy_calculator(qa_list: List[Dict[str, str]], response_list: List[str]) -> Tuple[str, List[bool]]:
    """Calculate accuracy by checking if correct answers are substrings of responses."""
    matches = []
    for i, qa_item in enumerate(qa_list):
        matched = qa_item["answer"] in response_list[i]
        matches.append(matched)
    
    accuracy = sum(matches) / len(matches)
    return f"Accuracy: {accuracy:.3f}", matches


def evaluate_enhanced_system(qa_list: List[Dict[str, str]], qa_chain, description: str) -> pd.DataFrame:
    """Evaluate the enhanced QA system."""
    results = {
        "question": [],
        "correct_answer": [],
        "context": [],
        "llm_answer": [],
        "ensemble_details": []
    }
    
    print(f"Evaluating {description} with {len(qa_list)} questions...")
    for qa in tqdm(qa_list, desc=f"Processing {description}"):
        result = qa_chain(qa["question"])
        
        results["question"].append(qa["question"])
        results["correct_answer"].append(qa["answer"])
        results["context"].append(result["context"])
        results["llm_answer"].append(result["answer"])
        results["ensemble_details"].append(result.get("ensemble_details", []))
    
    return pd.DataFrame(results)


def run_enhanced_evaluation():
    """Run enhanced evaluation with all improvements."""
    print("🚀" * 40)
    print("ENHANCED QA SYSTEM - PUSHING TO 85%+ ACCURACY!")
    print("🚀" * 40)
    
    # Load data
    qa_list = load_qa_data(QA_LIST_PATH)
    vector_db = load_vector_database(VECTOR_DB_PATH)
    
    # Test on substantial subset first
    test_subset = qa_list[:100]  # 50% of dataset
    
    print(f"\nTesting enhanced system on {len(test_subset)} questions...")
    print("Improvements implemented:")
    print("✅ Enhanced prompt with extraction examples")
    print("✅ Answer post-processing and cleaning")
    print("✅ Ensemble method with k=[2,3,4]")
    print("✅ Verbose answer truncation")
    print("✅ Better 'No Answer' detection")
    
    # Create enhanced system
    enhanced_chain = create_ensemble_qa_chain(vector_db, k_values=[2, 3, 4])
    
    # Evaluate
    start_time = time.time()
    results = evaluate_enhanced_system(test_subset, enhanced_chain, "ENHANCED SYSTEM")
    end_time = time.time()
    
    # Calculate accuracy
    accuracy_text, matches = accuracy_calculator(test_subset, results['llm_answer'].tolist())
    accuracy = float(accuracy_text.split(': ')[1])
    
    # Results
    print(f"\n🎯 ENHANCED SYSTEM RESULTS:")
    print(f"Accuracy: {accuracy:.3f} ({sum(matches)}/{len(matches)})")
    print(f"Time: {end_time - start_time:.1f}s ({(end_time - start_time)/len(test_subset):.2f}s/question)")
    
    # Compare to previous best
    previous_best = 0.77  # From full dataset test
    improvement = accuracy - previous_best
    
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"Previous best: {previous_best:.3f}")
    print(f"Enhanced system: {accuracy:.3f}")
    print(f"Improvement: +{improvement:.3f} ({improvement*100:.1f} percentage points)")
    
    if accuracy >= 0.85:
        print(f"🎉 TARGET ACHIEVED! {accuracy:.3f} exceeds 85% target!")
    elif accuracy >= 0.80:
        print(f"🎯 EXCELLENT! {accuracy:.3f} exceeds original 80% target!")
    else:
        points_needed = 0.85 - accuracy
        print(f"📊 Progress made! Need {points_needed:.3f} more points to reach 85%")
    
    # Error analysis on failures
    wrong_answers = []
    for i, (qa, match) in enumerate(zip(test_subset, matches)):
        if not match:
            wrong_answers.append({
                'question': qa['question'],
                'expected': qa['answer'],
                'got': results.iloc[i]['llm_answer'],
                'context_has_answer': qa['answer'] in results.iloc[i]['context']
            })
    
    if wrong_answers:
        print(f"\n🔍 ERROR ANALYSIS ({len(wrong_answers)} errors):")
        context_has_answer = sum(1 for w in wrong_answers if w['context_has_answer'])
        print(f"Errors where answer IS in context: {context_has_answer}/{len(wrong_answers)} ({context_has_answer/len(wrong_answers)*100:.1f}%)")
        
        print(f"\nSample errors:")
        for i, error in enumerate(wrong_answers[:3]):
            print(f"\n{i+1}. Q: {error['question'][:60]}...")
            print(f"   Expected: '{error['expected']}'")
            print(f"   Got: '{error['got']}'")
            print(f"   In context: {'✅' if error['context_has_answer'] else '❌'}")
    
    # Save results
    results.to_csv('enhanced_system_results.csv', index=False)
    
    detailed_results = {
        'accuracy': accuracy,
        'correct_count': sum(matches),
        'total_count': len(matches),
        'time': end_time - start_time,
        'improvement_over_baseline': improvement,
        'target_85_achieved': accuracy >= 0.85,
        'errors': wrong_answers[:10]  # Save first 10 errors
    }
    
    with open('enhanced_system_summary.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   • enhanced_system_results.csv")
    print(f"   • enhanced_system_summary.json")
    
    return accuracy, results


if __name__ == "__main__":
    accuracy, results = run_enhanced_evaluation()