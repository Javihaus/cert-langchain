"""
Example: Evaluate a RAG pipeline with CERTGroundingEvaluator in LangSmith.

Prerequisites:
    pip install "langchain-cert[local]" langchain langchain-openai

Usage:
    export LANGCHAIN_API_KEY=...
    export OPENAI_API_KEY=...
    python examples/langsmith_evaluation.py
"""

from langchain_cert import CERTGroundingEvaluator


def rag_pipeline(inputs: dict) -> dict:
    """
    Stub RAG pipeline. Replace with your actual implementation.
    Expected input keys: "input" or "question".
    Expected output keys: "output" or "answer".
    Optionally include "context" for SGI evaluation.
    """
    return {
        "output": f"Answer to: {inputs.get('input', '')}",
        "context": inputs.get("context", ""),
    }


def main():
    try:
        from langsmith.evaluation import evaluate
    except ImportError:
        print("Install langsmith: pip install langsmith")
        return

    evaluator = CERTGroundingEvaluator(
        # api_key="cert_xxx",  # optional: log to CERT dashboard
        threshold=0.45,         # normalized score threshold for pass/fail
    )

    print("CERTGroundingEvaluator initialized.")
    print(f"  SGI review threshold: 0.95 (raw), ~0.45 (normalized)")
    print(f"  DGI pass threshold:   0.30 (raw)")
    print()
    print("To run a full LangSmith evaluation:")
    print("  results = evaluate(rag_pipeline, data='your-dataset',")
    print("                     evaluators=[evaluator])")
    print("  evaluator.close()")


if __name__ == "__main__":
    main()
