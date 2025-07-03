from optimize_accuracy import load_qa_data, load_vector_database, create_qa_chain, evaluate_qa_system, accuracy_calculator
import time

def fine_tune_k_parameter():
    """Fine-tune k parameter around the optimal value."""
    print("Fine-tuning k parameter around optimal value...")
    
    qa_list = load_qa_data("qa_list.json")
    vector_db = load_vector_database("faiss_index")
    
    # Test with larger subset
    test_subset = qa_list[:100]  # Half the dataset
    
    k_values = [1, 2, 3, 4, 5]  # Fine-tune around k=3
    
    results = {}
    best_accuracy = 0
    best_k = 3
    
    for k in k_values:
        print(f"\nTesting k={k} with {len(test_subset)} questions")
        qa_chain = create_qa_chain(vector_db, k=k)
        
        start_time = time.time()
        eval_results = evaluate_qa_system(test_subset, qa_chain)
        end_time = time.time()
        
        accuracy_text, matches = accuracy_calculator(test_subset, eval_results['llm_answer'].tolist())
        accuracy = float(accuracy_text.split(': ')[1])
        
        results[k] = {
            'accuracy': accuracy,
            'time': end_time - start_time,
            'correct_count': sum(matches),
            'total_count': len(matches)
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
        
        print(f"K={k}: Accuracy={accuracy:.3f} ({sum(matches)}/{len(matches)}), Time={end_time-start_time:.1f}s")
    
    print(f"\n" + "="*50)
    print(f"OPTIMAL RESULTS:")
    print(f"Best k value: {best_k}")
    print(f"Best accuracy: {best_accuracy:.3f}")
    print(f"Improvement from k=10: {(best_accuracy - 0.79)*100:.1f} percentage points")
    print(f"="*50)
    
    return results, best_k

if __name__ == "__main__":
    results, best_k = fine_tune_k_parameter()