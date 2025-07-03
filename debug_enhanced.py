import json
import pandas as pd

# Load the results to debug
with open('enhanced_system_summary.json', 'r') as f:
    results = json.load(f)

print("🔍 DEBUGGING ENHANCED SYSTEM PERFORMANCE")
print("="*60)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Errors: {len(results['errors'])}")

print("\nSample errors:")
for i, error in enumerate(results['errors'][:5]):
    print(f"\n{i+1}. Expected: '{error['expected']}'")
    print(f"   Got: '{error['got']}'")
    print(f"   Analysis: ", end="")
    
    expected = error['expected']
    got = error['got']
    
    if expected in got:
        print("✅ Expected answer IS in LLM response - extraction too aggressive")
    elif got in expected:
        print("⚠️ LLM answer is subset of expected - extraction too conservative")
    else:
        print("❌ Completely different answers")

print(f"\n🔧 ISSUE IDENTIFIED:")
print("The enhanced cleaning is being too aggressive!")
print("We need to balance cleaning with preserving the right answer.")

# Quick fix strategy
print(f"\n💡 QUICK FIX STRATEGY:")
print("1. Reduce answer cleaning aggressiveness")
print("2. Focus on exact substring matching rather than cleaning")
print("3. Use simpler ensemble (just k=3 but with better prompt)")
print("4. Keep the improved prompt but reduce post-processing")