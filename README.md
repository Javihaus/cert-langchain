<div align="center">

![License](https://img.shields.io/badge/license-Apache%202.0-BB3381)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB)
![PyPI](https://img.shields.io/pypi/v/langchain-cert?color=4FB3B3)
[![CI](https://github.com/Javihaus/cert-langchain/actions/workflows/ci.yml/badge.svg)](https://github.com/Javihaus/cert-langchain/actions/workflows/ci.yml)

</div>

# Langchain-Cert

CERT hallucination detection for [LangChain](https://langchain.com) and [LangSmith](https://smith.langchain.com).

Detects LLM hallucinations using embedding geometry — no second LLM, no external API calls for scoring.

**With context (RAG):** Semantic Grounding Index -SGI- measures whether the response engaged with the source document.  
**Without context:** Directional Grounding Index -DGI- measures whether the response displacement vector aligns with grounded patterns.

[Cert Dashboard](https://cert-framework.com)

## Research

- [A Geometric Taxonomy of Hallucinations in LLMs](https://arxiv.org/pdf/2602.13224)
- [How Transformers Reject Wrong Answers: Rotational Dynamics of Factual Constraint Processing](https://arxiv.org/abs/2603.13259)
- [Semantic Grounding Index: Geometric Bounds on Context Engagement in RAG Systems](https://arxiv.org/abs/2512.13771)

---

## Installation

```bash
# Evaluator + local scoring
pip install "langchain-cert[local]"

# With CERT dashboard logging
pip install "langchain-cert[all]"
```

## LangSmith Evaluation

```python
from langchain_cert import CERTGroundingEvaluator
from langsmith.evaluation import evaluate

evaluator = CERTGroundingEvaluator()

results = evaluate(
    your_pipeline,
    data="your-langsmith-dataset",
    evaluators=[evaluator],
)
evaluator.close()  # flush pending dashboard traces
```

With dashboard logging:

```python
evaluator = CERTGroundingEvaluator(
    api_key="cert_xxx",
    project="production",
)
```

## LangChain Callback

Logs every LLM call to CERT for async evaluation — no pipeline changes required.

```python
from langchain_cert import CERTCallbackHandler
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    callbacks=[CERTCallbackHandler(api_key="cert_xxx", project="prod")]
)
```

## How it works

### SGI — Semantic Grounding Index (arXiv:2512.13771)

```
SGI = dist(response, question) / dist(response, context)
```

High SGI: response moved toward context (grounded).  
Low SGI: response stayed near the question (semantic laziness, confabulation risk).

| SGI | Interpretation |
|-----|----------------|
| >= 1.20 | Strong context engagement |
| 0.95 – 1.20 | Partial engagement, review recommended |
| < 0.95 | Confabulation risk |

### DGI — Directional Grounding Index (arXiv:2602.13224)

```
DGI = dot(normalize(phi(r) - phi(q)), mu_hat)
```

Measures whether the query-to-response displacement aligns with a reference
direction computed from verified grounded pairs. Detects confabulation
without source documents.

| DGI | Interpretation |
|-----|----------------|
| >= 0.30 | Displacement matches grounded patterns |
| < 0.30 | Unusual displacement, review recommended |

#### Calibration

DGI requires a reference direction — a mean displacement vector computed from
known-grounded (question, response) pairs. Calibration quality determines
detection accuracy:

| Calibration | Dataset | Typical AUROC |
|-------------|---------|---------------|
| Generic (bundled) | Finance, medical, science, law, coding, and general domains (212 pairs) | ~0.76 |
| Domain-specific | Your verified production pairs | 0.90 – 0.99 |

The bundled dataset contains finance and medical domain pairs. It works
out of the box but degrades on unrelated domains — this is architectural,
not a bug. See [arXiv:2602.13224](https://arxiv.org/abs/2602.13224) §3.2
for the fiber bundle structure that explains why.

#### Domain-specific calibration

Provide a CSV of verified grounded pairs from your domain:

```python
evaluator = CERTGroundingEvaluator(
    reference_csv="my_pairs.csv"
)
```

CSV format — comma or semicolon delimited, header row required:

```csv
question,response
What is our refund policy?,Refunds are processed within 5 business days.
How do I reset my password?,Navigate to Settings > Security > Reset Password.
```

Required columns: `question` and one of `response`, `answer`, or `output`.

Include only verified grounded responses. Hallucinated responses in the
reference set will degrade DGI accuracy.

For automatic domain calibration with optimization across your full trace
history, use the [CERT dashboard](https://cert-framework.com).

### What CERT cannot detect

Type III hallucinations — factually wrong responses within the correct semantic frame —
are geometrically undetectable. Embeddings encode distributional co-occurrence, not
truth correspondence. CERT measures whether the model *engaged with context*, not
whether the facts are correct. See arXiv:2602.13224 §4.3 for the formal treatment.

## Scores in LangSmith

Each evaluated run receives a `cert_grounding` score in [0, 1]:

- `1.0` — strongly grounded
- `0.5` — at the review boundary
- `0.0` — ungrounded

The `comment` field includes the raw SGI/DGI value, thresholds, and method.

## License

Apache 2.0.
