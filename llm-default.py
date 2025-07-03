import json
import os
import warnings
from typing import List, Dict, Tuple, Any

import pandas as pd
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langsmith.utils import LangSmithMissingAPIKeyWarning
from dotenv import load_dotenv

# Configuration
QA_LIST_PATH = "qa_list.json"
VECTOR_DB_PATH = "faiss_index"
RETRIEVAL_K = 3  # Optimized: reduced from 10 to 3 for better accuracy
LLM_MODEL = "gpt-4o-mini"

# Load environment variables
load_dotenv()

# Suppress LangSmith warnings
warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)


def load_qa_data(file_path: str) -> List[Dict[str, str]]:
    """Load question-answer pairs from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"QA data file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")


def load_vector_database(db_path: str) -> FAISS:
    """Load pre-built FAISS vector database."""
    embeddings = OpenAIEmbeddings()
    try:
        vector_db = FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True
        )
        print(f"Successfully loaded vector database from '{db_path}'")
        return vector_db
    except Exception as e:
        raise RuntimeError(f"Failed to load vector database: {e}")


def accuracy_calculator(qa_list: List[Dict[str, str]], response_list: List[str]) -> Tuple[str, List[bool]]:
    """Calculate accuracy by checking if correct answers are substrings of responses.
    
    Args:
        qa_list: List of question-answer dictionaries
        response_list: List of LLM responses
        
    Returns:
        Tuple of (accuracy_string, list_of_matches)
    """
    matches = []
    for i, qa_item in enumerate(qa_list):
        matched = qa_item["answer"] in response_list[i]
        matches.append(matched)
    
    accuracy = sum(matches) / len(matches)
    return f"Accuracy: {accuracy:.2f}", matches


def create_qa_prompt() -> ChatPromptTemplate:
    """Create the optimized prompt template for question answering.
    
    This custom prompt significantly outperforms the standard LangChain Hub 
    RAG prompt (88% vs 42% accuracy) due to explicit exact matching instructions.
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


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vector_db: FAISS, k: int = RETRIEVAL_K, model: str = LLM_MODEL):
    """Create the question-answering chain."""
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    prompt = create_qa_prompt()
    llm_chain = prompt | ChatOpenAI(model=model) | StrOutputParser()
    
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


def evaluate_qa_system(qa_list: List[Dict[str, str]], qa_chain) -> pd.DataFrame:
    """Evaluate the QA system on the provided dataset."""
    results = {
        "question": [],
        "correct_answer": [],
        "context": [],
        "llm_answer": [],
    }
    
    print(f"Evaluating {len(qa_list)} questions...")
    for qa in tqdm(qa_list, desc="Processing questions"):
        result = qa_chain.invoke(qa["question"])
        results["question"].append(qa["question"])
        results["correct_answer"].append(qa["answer"])
        results["context"].append(result["context"])
        results["llm_answer"].append(result["answer"])
    
    # Calculate accuracy
    accuracy_text, matches = accuracy_calculator(qa_list, results["llm_answer"])
    results["is_correct"] = matches
    
    print(f"\n{accuracy_text}")
    return pd.DataFrame(results)


def main():
    """Main execution function."""
    try:
        # Load data and models
        qa_list = load_qa_data(QA_LIST_PATH)
        vector_db = load_vector_database(VECTOR_DB_PATH)
        
        # Create QA chain
        qa_chain = create_qa_chain(vector_db)
        
        # Evaluate system
        results_df = evaluate_qa_system(qa_list, qa_chain)
        
        # Optional: Save results
        # results_df.to_csv("qa_results.csv", index=False)
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
