# Quick Start - RAG Retrieval Pipeline

Master RAG retrieval strategies in under 2 minutes!

## What You'll Learn

How to retrieve the most relevant information from a vector database using:

- Similarity search with different k values
- Relevance scores
- MMR for diverse results
- Retriever interface
- Quality analysis

**Note**: This demo focuses on RETRIEVAL only. For complete RAG with LLM generation, see demo-11.

## 60-Second Setup

```bash
# 1. Navigate
cd demo-10-rag-retrieval-pipeline

# 2. Setup
uv venv && source .venv/bin/activate
uv pip install -e .

# 3. Configure
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-your-key

# 4. Run!
uv run python main.py
```

## Configuration

### Minimal (ChromaDB - Local)

```bash
# .env
OPENAI_API_KEY=sk-...
VECTOR_DB=chromadb
```

### Cloud (Pinecone)

```bash
# .env
OPENAI_API_KEY=sk-...
VECTOR_DB=pinecone
PINECONE_API_KEY=your-key
```

## What Happens

```
1. Ingestion Phase
   ├── Load documents (PDF, text, web)
   ├── Chunk into 1000-char pieces
   └── Store with embeddings

2. Retrieval Demonstrations (6 Scenarios)
   ├── [1] Different k values (k=2, 4, 6)
   ├── [2] Relevance scores
   ├── [3] MMR for diversity
   ├── [4] Retriever interface
   ├── [5] Quality analysis
   └── [6] Document inspection
```

## Understanding Output

### Ingestion

```
✓ Loaded 8 documents
✓ Created 127 chunks
✓ Stored with embeddings in CHROMADB
```

### Retrieval Scenarios

**[1] Different K Values**

```
[k=2] Retrieved 2 documents
  [1] Documents/company_policy.pdf
  [2] Documents/guidelines.txt

[k=4] Retrieved 4 documents
  (More results for broader coverage)
```

**[2] Relevance Scores**

```
✓ Retrieved 3 documents with scores
  [1] Score: 0.8542 | Very relevant
  [2] Score: 0.7234 | Moderately relevant
  [3] Score: 0.6891 | Less relevant
```

**[3] MMR Search**

```
✓ Retrieved 4 diverse documents
  (Avoids redundant similar results)
```

**[4] Retriever Interface**

```
✓ Using standard LangChain retriever
✓ Returns 4 documents
```

**[5] Quality Analysis**

```
--- k=1 ---
  Average score: 0.8542

--- k=3 ---
  Average score: 0.7556
  (More results = lower average relevance)
```

**[6] Document Details**

```
======================================
DOCUMENT #1 DETAILS
======================================
[Metadata]
  source: company_policy.pdf
  page: 4

[Content]
  892 characters
  Preview: Employee benefits include...
```

## Key Concepts (30 Seconds)

### Similarity Search

Find documents most similar to query.

```python
results = vectorstore.similarity_search(query, k=3)
```

### K Value

Number of results to retrieve.

- Low k (1-3): Precision
- Medium k (4-6): Balance ⭐
- High k (10+): Recall

### Relevance Score

How similar is each result (0-1).

- Lower score = More similar
- 0.8+ = Very relevant
- 0.6-0.8 = Moderately relevant
- < 0.6 = Less relevant

### MMR (Maximum Marginal Relevance)

Balance relevance with diversity.

```python
# Avoid redundant results
results = vectorstore.max_marginal_relevance_search(
    query, k=4, fetch_k=10
)
```

## Quick Experiments

### 1. Try Different Queries

Edit bottom of `main.py`:

```python
# Around line 440
query1 = "Your question here?"
```

### 2. Adjust K Values

```python
# Line ~380
for k in [1, 3, 5, 10]:  # Try different ranges
```

### 3. Change Chunk Size

```python
# Line ~54
CHUNK_SIZE = 500   # Smaller chunks
CHUNK_SIZE = 1500  # Larger chunks
```

### 4. Add Your Documents

```bash
cp ~/my-docs/*.pdf Documents/
cp ~/my-docs/*.txt Documents/
uv run python main.py
```

## Retrieval Strategies Quick Reference

| Strategy        | When to Use            | Command                                     |
| --------------- | ---------------------- | ------------------------------------------- |
| **Similarity**  | Most relevant results  | `similarity_search(query, k=3)`             |
| **With Scores** | Need relevance metrics | `similarity_search_with_score(query, k=3)`  |
| **MMR**         | Avoid redundancy       | `max_marginal_relevance_search(query, k=4)` |
| **Retriever**   | Standard interface     | `retriever.invoke(query)`                   |
| **Filtered**    | Specific documents     | `search_kwargs={"filter": {...}}`           |

## Common Tasks

### See More/Fewer Results

```python
# More results (broader coverage)
results = vectorstore.similarity_search(query, k=10)

# Fewer results (only top matches)
results = vectorstore.similarity_search(query, k=2)
```

### Get Relevance Scores

```python
results = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in results:
    print(f"Score: {score:.4f}")
```

### Get Diverse Results

```python
# Fetch 20 candidates, return 5 diverse ones
results = vectorstore.max_marginal_relevance_search(
    query, k=5, fetch_k=20
)
```

### Use Standard Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
results = retriever.invoke(query)
```

## Understanding K Value

```
k=1  →  [Best Match]
        High precision, might miss relevant info

k=3  →  [Top 3 Matches]
        Good balance ⭐

k=5  →  [Top 5 Matches]
        More coverage

k=10 →  [Top 10 Matches]
        Maximum coverage, lower average quality
```

## File Structure

```
demo-10-rag-retrieval-pipeline/
├── main.py          # ~450 lines, retrieval-focused
├── Documents/       # Your knowledge base
├── .env            # Configuration
└── README.md       # Full documentation
```

## Switching Vector Databases

```bash
# Local (ChromaDB) - Zero setup
VECTOR_DB=chromadb

# Cloud (Pinecone) - Requires API key
VECTOR_DB=pinecone
PINECONE_API_KEY=your-key
```

## Troubleshooting

### "No documents loaded"

```bash
# Create Documents folder and add files
mkdir -p Documents
cp ~/my-docs/*.pdf Documents/
cp ~/my-docs/*.txt Documents/
```

### "No results found"

- Vector database might be empty
- Try broader queries
- Lower your k value
- Check if documents actually loaded

### "MMR not supported"

- Some vector stores don't support MMR
- This is normal, demo handles gracefully
- Use similarity search instead

### Want fresh start?

```bash
rm -rf chroma_db/
uv run python main.py
```

## Performance Tips

### Chunk Size Impact

```python
# Smaller chunks (500 chars)
+ More precise retrieval
- Might miss context

# Larger chunks (1500 chars)
+ More context per result
- Less precise matching
```

### K Value Impact

```python
# Small k (1-2)
+ High precision
+ Faster
- Might miss relevant info

# Large k (10+)
+ More coverage
- Lower average quality
- Slower
```

### Relevance Score Thresholds

```python
# Only high-quality results
if score > 0.8:
    # Very relevant

# Moderate quality acceptable
if score > 0.6:
    # Moderately relevant
```

## What's NOT in This Demo

❌ **LLM-based answer generation** (see demo-11)  
❌ **Conversational memory** (see demo-12)  
❌ **Streaming responses** (see demo-13)

This demo focuses PURELY on retrieval strategies.

## Next Steps

1. **Run as-is** - See all retrieval strategies
2. **Try queries** - Test with your questions
3. **Adjust k** - Find optimal number of results
4. **Change chunks** - See impact on quality
5. **Add documents** - Use your own data
6. **Try MMR** - Compare with similarity search
7. **Move to demo-11** - Add LLM generation

## Why Retrieval-Only?

Understanding retrieval is crucial because:

- ✓ Better retrieval = Better RAG answers
- ✓ You can tune retrieval independently
- ✓ Retrieval quality is measurable
- ✓ Different strategies for different needs
- ✓ Foundation for advanced RAG techniques

## Learning Outcomes

After this demo, you'll understand:

- ✓ How similarity search works
- ✓ What relevance scores mean
- ✓ When to use different k values
- ✓ How MMR provides diversity
- ✓ How to use retriever interface
- ✓ How to analyze retrieval quality

## Quick Reference Card

```bash
# Run demo
uv run python main.py

# Clean database
rm -rf chroma_db/

# Add documents
cp *.pdf Documents/

# Switch to Pinecone
# Edit .env: VECTOR_DB=pinecone

# Check file size
wc -l main.py  # ~450 lines
```

## Comparison

| Demo        | Focus          | Output                |
| ----------- | -------------- | --------------------- |
| **demo-09** | Ingestion only | Stores documents      |
| **demo-10** | Retrieval only | Finds relevant chunks |
| **demo-11** | Complete RAG   | Generates answers     |

---

**Setup time**: 2 minutes  
**Strategies shown**: 6  
**Lines of code**: ~450 (single file)  
**Output**: Comparison of retrieval strategies

Ready? `uv run python main.py` 🚀
