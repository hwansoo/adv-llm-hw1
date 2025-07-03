# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Question-Answering (QA) system that uses LangChain with OpenAI embeddings and FAISS vector database for retrieval-augmented generation (RAG). The system evaluates QA performance by comparing LLM responses against expected answers.

## Project Objectives

- **Target Accuracy**: Achieve accuracy higher than 0.8 (80%)
- **Token Efficiency**: Minimize token usage during testing to reduce costs
- **Fixed Constraints**: 
  - FAISS index must be used as-is (no modifications allowed)
  - Must use OpenAI LLM for generation
  - Accuracy evaluation must use the existing `accuracy_calculator()` function (exact substring matching)

## Architecture

- **Main Script**: `llm-default.py` - The core QA evaluation system
- **Vector Database**: Pre-built FAISS index stored in `faiss_index/` directory
- **Test Data**: `qa_list.json` - Contains 800 question-answer pairs for evaluation
- **Configuration**: `.env` file contains OpenAI API key

## Key Components

### RAG Pipeline
- Uses OpenAI embeddings for text vectorization
- FAISS vector store for similarity search (k=10 retrieval)
- GPT-4o-mini as the language model
- Custom prompt template ensuring exact substring matching from context

### Evaluation System
- `accuracy_calculator()` function compares LLM answers against ground truth
- Exact substring matching for accuracy computation
- Results stored in pandas DataFrame with detailed breakdown

## Running the System

```bash
# Ensure Python 3.12+ is available
python3 llm-default.py
```

## Dependencies

The system requires:
- `langchain-openai` - OpenAI integration
- `langchain-community` - FAISS vector store
- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `pandas` - Data manipulation
- `tqdm` - Progress bars
- `numpy` - Numerical operations

## Environment Setup

1. Set OpenAI API key in `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. The FAISS index must be pre-built and stored in `faiss_index/` directory containing:
   - `index.faiss` - The FAISS index file
   - `index.pkl` - Metadata pickle file

## Optimization Strategies

To achieve the target accuracy while minimizing token usage:

1. **Prompt Engineering**: Refine the prompt template to improve answer extraction accuracy
2. **Model Selection**: Consider using more cost-effective OpenAI models (gpt-3.5-turbo vs gpt-4o-mini)
3. **Retrieval Parameters**: Experiment with different k values for context retrieval
4. **Context Processing**: Optimize context formatting to reduce token count while maintaining relevance

## Important Notes

- The system uses `allow_dangerous_deserialization=True` when loading FAISS index (required for LangChain v0.2.0+)
- Prompt is designed for exact substring matching from retrieved context
- No installation requirements file present - dependencies must be installed manually
- Vector database is pre-built and not created by the main script
- Current baseline accuracy can be measured by running the existing system