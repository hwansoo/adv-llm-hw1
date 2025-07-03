import json
import pandas as pd
from typing import List, Dict
import re

def analyze_wrong_answers():
    """Analyze patterns in wrong answers to identify improvement opportunities."""
    
    # Load the results
    with open('full_dataset_results.json', 'r') as f:
        results = json.load(f)
    
    # Load the QA data and run a quick evaluation to get individual results
    with open('qa_list.json', 'r') as f:
        qa_list = json.load(f)
    
    print("="*80)
    print("ERROR ANALYSIS: Understanding Wrong Answers")
    print("="*80)
    
    # Run a quick sample to get some wrong answers for analysis
    from full_dataset_test import load_vector_database, create_optimized_chain, format_docs
    
    vector_db = load_vector_database("faiss_index")
    optimized_chain = create_optimized_chain(vector_db, k=3)
    
    print("Analyzing a sample of questions to identify error patterns...")
    
    # Test first 30 questions to get some errors
    sample_size = 30
    wrong_answers = []
    correct_answers = []
    
    for i, qa in enumerate(qa_list[:sample_size]):
        result = optimized_chain.invoke(qa["question"])
        
        if isinstance(result, dict):
            llm_answer = result.get("answer", str(result))
            context = result.get("context", "")
        else:
            llm_answer = str(result)
            context = ""
        
        is_correct = qa["answer"] in llm_answer
        
        analysis_item = {
            'question': qa["question"],
            'correct_answer': qa["answer"],
            'llm_answer': llm_answer,
            'context': context,
            'is_correct': is_correct,
            'correct_answer_length': len(qa["answer"].split()),
            'llm_answer_length': len(llm_answer.split()),
            'is_no_answer': qa["answer"] == "No Answer",
            'llm_said_no_answer': "No Answer" in llm_answer
        }
        
        if is_correct:
            correct_answers.append(analysis_item)
        else:
            wrong_answers.append(analysis_item)
    
    print(f"\nSample Analysis Results:")
    print(f"Correct: {len(correct_answers)}/{sample_size} ({len(correct_answers)/sample_size*100:.1f}%)")
    print(f"Wrong: {len(wrong_answers)}/{sample_size} ({len(wrong_answers)/sample_size*100:.1f}%)")
    
    # Analyze error patterns
    if wrong_answers:
        print(f"\n" + "="*60)
        print("ERROR PATTERN ANALYSIS")
        print("="*60)
        
        # Pattern 1: "No Answer" errors
        correct_no_answer = sum(1 for w in wrong_answers if w['is_no_answer'] and w['llm_said_no_answer'])
        incorrect_no_answer = sum(1 for w in wrong_answers if w['is_no_answer'] and not w['llm_said_no_answer'])
        false_no_answer = sum(1 for w in wrong_answers if not w['is_no_answer'] and w['llm_said_no_answer'])
        
        print(f"\n1. 'No Answer' Pattern Analysis:")
        print(f"   • Should be 'No Answer' but LLM gave answer: {incorrect_no_answer}")
        print(f"   • Shouldn't be 'No Answer' but LLM said 'No Answer': {false_no_answer}")
        
        # Pattern 2: Answer length analysis
        avg_correct_length = sum(w['correct_answer_length'] for w in wrong_answers) / len(wrong_answers)
        avg_llm_length = sum(w['llm_answer_length'] for w in wrong_answers) / len(wrong_answers)
        
        print(f"\n2. Answer Length Analysis:")
        print(f"   • Average correct answer length: {avg_correct_length:.1f} words")
        print(f"   • Average LLM answer length: {avg_llm_length:.1f} words")
        
        # Pattern 3: Show specific examples
        print(f"\n3. Sample Wrong Answers:")
        for i, error in enumerate(wrong_answers[:5]):
            print(f"\n   Example {i+1}:")
            print(f"   Question: {error['question'][:80]}...")
            print(f"   Expected: '{error['correct_answer']}'")
            print(f"   Got: '{error['llm_answer'][:100]}...'")
            if error['correct_answer'] in error['context']:
                print(f"   ✓ Correct answer IS in context")
            else:
                print(f"   ✗ Correct answer NOT in context")
        
        # Pattern 4: Context analysis
        answers_in_context = sum(1 for w in wrong_answers if w['correct_answer'] in w['context'])
        print(f"\n4. Context Analysis:")
        print(f"   • Wrong answers where correct answer IS in context: {answers_in_context}/{len(wrong_answers)}")
        print(f"   • This suggests prompt/extraction issues rather than retrieval issues")
    
    # Analyze correct answers for comparison
    if correct_answers:
        print(f"\n" + "="*60)
        print("SUCCESSFUL PATTERN ANALYSIS")
        print("="*60)
        
        avg_correct_length_success = sum(c['correct_answer_length'] for c in correct_answers) / len(correct_answers)
        no_answer_success = sum(1 for c in correct_answers if c['is_no_answer'])
        
        print(f"Average successful answer length: {avg_correct_length_success:.1f} words")
        print(f"Successful 'No Answer' cases: {no_answer_success}")
    
    print(f"\n" + "="*80)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("="*80)
    
    print("\n🎯 HIGH IMPACT IMPROVEMENTS:")
    print("\n1. **Answer Post-Processing & Validation**")
    print("   • Clean extracted answers (remove extra text)")
    print("   • Validate answer format and length")
    print("   • Better 'No Answer' detection logic")
    
    print("\n2. **Enhanced Prompt Engineering**")
    print("   • More specific extraction instructions")
    print("   • Examples of correct vs incorrect extractions")
    print("   • Stronger emphasis on exact substring matching")
    
    print("\n3. **Context Quality Improvement**")
    print("   • Pre-filter retrieved documents for relevance")
    print("   • Remove noisy/irrelevant sentences")
    print("   • Highlight key phrases in context")
    
    print("\n🔧 MEDIUM IMPACT IMPROVEMENTS:")
    print("\n4. **Ensemble Methods**")
    print("   • Combine predictions from multiple k values (k=2, k=3, k=4)")
    print("   • Vote or confidence-weight different approaches")
    print("   • Use multiple prompt variations")
    
    print("\n5. **Retrieval Enhancement**")
    print("   • Rerank retrieved documents by relevance")
    print("   • Use query expansion for better matching")
    print("   • Filter by document type/metadata")
    
    print("\n6. **Answer Classification**")
    print("   • Separate handling for different answer types")
    print("   • Special logic for dates, names, numbers")
    print("   • Custom handling for 'No Answer' cases")
    
    print("\n🚀 ADVANCED IMPROVEMENTS:")
    print("\n7. **Model Upgrades**")
    print("   • Test GPT-4 for higher capability (cost consideration)")
    print("   • Fine-tune embeddings on this domain")
    print("   • Use specialized QA models")
    
    print("\n8. **Dynamic Context Management**")
    print("   • Adaptive k values based on question type")
    print("   • Context compression and summarization")
    print("   • Multi-step reasoning for complex questions")
    
    print(f"\n" + "="*80)
    print("REALISTIC TARGET: 85-90% Accuracy")
    print("="*80)
    print("Current: 77% → Target: 85%+ (8+ point improvement needed)")
    print("\nPriority implementation order:")
    print("1. Answer post-processing (quick win)")
    print("2. Enhanced prompt with examples")
    print("3. Context quality filtering")
    print("4. Ensemble methods for hard cases")

if __name__ == "__main__":
    analyze_wrong_answers()