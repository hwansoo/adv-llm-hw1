import json
import os
import warnings
from typing import List, Dict, Tuple, Any
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


def accuracy_calculator(qa_list: List[Dict[str, str]], response_list: List[str]) -> Tuple[str, List[bool]]:
    """Calculate accuracy by checking if correct answers are substrings of responses."""
    matches = []
    for i, qa_item in enumerate(qa_list):
        matched = qa_item["answer"] in response_list[i]
        matches.append(matched)
    
    accuracy = sum(matches) / len(matches)
    return f"Accuracy: {accuracy:.2f}", matches


def evaluate_qa_system(qa_list: List[Dict[str, str]], qa_chain) -> pd.DataFrame:
    """Evaluate the QA system on the provided dataset."""
    results = {
        "question": [],
        "correct_answer": [],
        "context": [],
        "llm_answer": [],
    }
    
    for qa in tqdm(qa_list, desc="Processing questions"):
        result = qa_chain.invoke(qa["question"])
        results["question"].append(qa["question"])
        results["correct_answer"].append(qa["answer"])
        results["context"].append(result["context"])
        results["llm_answer"].append(result["answer"])
    
    return pd.DataFrame(results)


def create_optimized_prompt_v1() -> ChatPromptTemplate:
    """Improved prompt with better instructions for exact matching."""
    return ChatPromptTemplate.from_template(
        """
You are an expert at extracting precise answers from text. Your task is to find the exact answer in the given context.

Context:
{context}

CRITICAL INSTRUCTIONS:
1. The answer MUST be an exact substring from the context above
2. Copy the text exactly as it appears - preserve all punctuation, capitalization, and spacing
3. Do not add, remove, or modify any characters
4. If you cannot find the answer in the context, respond with exactly "No Answer"
5. Look for the most specific and complete answer that directly addresses the question

Question: {question}

Extract the exact answer from the context:
"""
    )


def create_optimized_prompt_v2() -> ChatPromptTemplate:
    """Alternative prompt focusing on substring extraction."""
    return ChatPromptTemplate.from_template(
        """
Extract the exact answer from the context. The answer must be a direct quote from the text.

Context:
{context}

Question: {question}

Instructions:
- Find the exact phrase in the context that answers the question
- Copy it character-for-character including punctuation
- If no answer exists in the context, respond "No Answer"
- Do not paraphrase or modify the text in any way

Answer:
"""
    )


def create_qa_chain(vector_db: FAISS, k: int = 10, model: str = "gpt-4o-mini", prompt_template=None):
    """Create the question-answering chain."""
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    
    if prompt_template is None:
        prompt_template = ChatPromptTemplate.from_template(
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
    
    llm_chain = prompt_template | ChatOpenAI(model=model) | StrOutputParser()
    
    return {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | RunnableLambda(
        lambda inputs: {
            "question": inputs["question"],
            "context": inputs["context"],
            "answer": llm_chain.invoke(
                {"context": inputs["context"], "question": inputs["question"]}
            ),
        }
    )


def test_retrieval_parameters():
    """Test different k values for retrieval."""
    print("Testing retrieval parameters...")
    
    qa_list = load_qa_data(QA_LIST_PATH)
    vector_db = load_vector_database(VECTOR_DB_PATH)
    
    # Test with smaller subset for speed
    test_subset = qa_list[:10]
    
    results = {}
    
    for k in [5, 10, 15, 20]:
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
        
        print(f"K={k}: Accuracy={accuracy:.2f}, Time={end_time-start_time:.1f}s")
    
    return results


def test_prompt_variations():
    """Test different prompt templates."""
    print("Testing prompt variations...")
    
    qa_list = load_qa_data(QA_LIST_PATH)
    vector_db = load_vector_database(VECTOR_DB_PATH)
    
    # Test with smaller subset for speed
    test_subset = qa_list[:10]
    
    prompts = {
        'original': None,  # Uses default from create_qa_chain
        'v1': create_optimized_prompt_v1(),
        'v2': create_optimized_prompt_v2()
    }
    
    results = {}
    
    for name, prompt_template in prompts.items():
        print(f"\nTesting prompt: {name}")
        
        qa_chain = create_qa_chain(vector_db, k=10, prompt_template=prompt_template)
        
        start_time = time.time()
        eval_results = evaluate_qa_system(test_subset, qa_chain)
        end_time = time.time()
        
        accuracy_text, _ = accuracy_calculator(test_subset, eval_results['llm_answer'].tolist())
        accuracy = float(accuracy_text.split(': ')[1])
        
        results[name] = {
            'accuracy': accuracy,
            'time': end_time - start_time
        }
        
        print(f"Prompt {name}: Accuracy={accuracy:.2f}, Time={end_time-start_time:.1f}s")
    
    return results


def test_model_variations():
    """Test different OpenAI models."""
    print("Testing model variations...")
    
    qa_list = load_qa_data(QA_LIST_PATH)
    vector_db = load_vector_database(VECTOR_DB_PATH)
    
    # Test with smaller subset for speed
    test_subset = qa_list[:5]  # Even smaller for expensive models
    
    models = ["gpt-4o-mini", "gpt-3.5-turbo"]  # Removed gpt-4o for cost
    
    results = {}
    
    for model in models:
        print(f"\nTesting model: {model}")
        
        try:
            qa_chain = create_qa_chain(vector_db, k=10, model=model)
            
            start_time = time.time()
            eval_results = evaluate_qa_system(test_subset, qa_chain)
            end_time = time.time()
            
            accuracy_text, _ = accuracy_calculator(test_subset, eval_results['llm_answer'].tolist())
            accuracy = float(accuracy_text.split(': ')[1])
            
            results[model] = {
                'accuracy': accuracy,
                'time': end_time - start_time
            }
            
            print(f"Model {model}: Accuracy={accuracy:.2f}, Time={end_time-start_time:.1f}s")
            
        except Exception as e:
            print(f"Error with model {model}: {e}")
            results[model] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    print("Starting optimization tests...")
    
    # Test just prompts first
    print("\n=== Testing Prompt Variations ===")
    prompt_results = test_prompt_variations()
    
    print(f"\nPrompt Results Summary:")
    for name, result in prompt_results.items():
        if 'accuracy' in result:
            print(f"{name}: {result['accuracy']:.2f} accuracy, {result['time']:.1f}s")
    
    # Save results
    with open('prompt_optimization_results.json', 'w') as f:
        json.dump(prompt_results, f, indent=2)