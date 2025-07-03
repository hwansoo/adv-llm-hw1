from llm_default import main
import time

def run_final_evaluation():
    """Run final evaluation on full dataset with optimized parameters."""
    print("=" * 60)
    print("FINAL EVALUATION - OPTIMIZED QA SYSTEM")
    print("=" * 60)
    print("Optimizations applied:")
    print("- Reduced retrieval parameter k from 10 to 3")
    print("- Using original prompt template (best performing)")
    print("- Model: gpt-4o-mini")
    print("=" * 60)
    
    start_time = time.time()
    main()
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"\nTotal evaluation time: {total_time:.1f} seconds")
    print(f"Average time per question: {total_time/200:.2f} seconds")

if __name__ == "__main__":
    run_final_evaluation()