from optimize_accuracy import load_qa_data, load_vector_database, create_qa_chain, evaluate_qa_system, accuracy_calculator
import time

def test_retrieval_detailed():
    """Test retrieval parameters in detail."""
    print("Testing retrieval parameters...")
    
    qa_list = load_qa_data("qa_list.json")
    vector_db = load_vector_database("faiss_index")
    
    # Test with slightly larger subset
    test_subset = qa_list[:20]
    
    results = {}
    
    for k in [3, 5, 8, 10, 15, 20, 25]:
        print(f"\nTesting k={k}")
        qa_chain = create_qa_chain(vector_db, k=k)
        
        start_time = time.time()
        eval_results = evaluate_qa_system(test_subset, qa_chain)
        end_time = time.time()
        
        accuracy_text, _ = accuracy_calculator(test_subset, eval_results['llm_answer'].tolist())
        accuracy = float(accuracy_text.split(': ')[1])
        
        results[k] = {
            'accuracy': accuracy,
            'time': end_time - start_time,
            'avg_time_per_question': (end_time - start_time) / len(test_subset)
        }
        
        print(f"K={k}: Accuracy={accuracy:.2f}, Time={end_time-start_time:.1f}s, Avg={results[k]['avg_time_per_question']:.1f}s/q")
    
    # Find best k
    best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print(f"\nBest k value: {best_k} with accuracy {results[best_k]['accuracy']:.2f}")
    
    return results, best_k

if __name__ == "__main__":
    results, best_k = test_retrieval_detailed()