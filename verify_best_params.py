from optimize_accuracy import load_qa_data, load_vector_database, create_qa_chain, evaluate_qa_system, accuracy_calculator
import time

def verify_optimal_k():
    """Verify k=3 with larger test set."""
    print("Verifying optimal k value with larger test set...")
    
    qa_list = load_qa_data("qa_list.json")
    vector_db = load_vector_database("faiss_index")
    
    # Test with larger subset to confirm
    test_subset = qa_list[:50]  # 25% of total
    
    k_values = [3, 5, 10]  # Test the promising ones
    
    results = {}
    
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
        
        print(f"K={k}: Accuracy={accuracy:.3f} ({sum(matches)}/{len(matches)}), Time={end_time-start_time:.1f}s")
    
    # Find best
    best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print(f"\nBest k value: {best_k}")
    print(f"Best accuracy: {results[best_k]['accuracy']:.3f}")
    print(f"Correct answers: {results[best_k]['correct_count']}/{results[best_k]['total_count']}")
    
    return results, best_k

if __name__ == "__main__":
    results, best_k = verify_optimal_k()