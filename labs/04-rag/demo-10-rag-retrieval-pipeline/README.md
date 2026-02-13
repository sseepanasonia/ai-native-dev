# Demo 10: RAG Retrieval Pipeline

This demo focuses on the **RETRIEVAL phase** of RAG (Retrieval-Augmented Generation). It demonstrates various retrieval strategies and scenarios for finding the most relevant information from a vector database.

## What This Demo Covers

**Focus**: Understanding retrieval quality and strategies  
**Does NOT include**: LLM-based answer generation

For complete RAG with generation, see **demo-11-complete-rag-pipeline**.

## What is RAG Retrieval?

RAG retrieval is the process of finding the most relevant pieces of information from a knowledge base to answer a query. It's the foundation of RAG systems.

### Retrieval Workflow

```
1. INGESTION PHASE (Setup)
   ┌─────────────┐
   │  Documents  │ → Load documents
   └─────────────┘
          ↓
   ┌─────────────┐
   │   Chunks    │ → Split into pieces
   └─────────────┘
          ↓
   ┌─────────────┐
   │ Embeddings  │ → Generate vectors
   └─────────────┘
          ↓
   ┌─────────────┐
   │ Vector DB   │ → Store for search
   └─────────────┘

2. RETRIEVAL PHASE (Query Time)
   ┌─────────────┐
   │   Query     │ → User's question
   └─────────────┘
          ↓
   ┌─────────────┐
   │  Embedding  │ → Convert to vector
   └─────────────┘
          ↓
   ┌─────────────┐
   │   Search    │ → Find similar chunks
   └─────────────┘
          ↓
   ┌─────────────┐
   │  Results    │ → Relevant documents
   └─────────────┘
```

## Key Features

✅ **Retrieval-Focused** - Deep dive into retrieval strategies  
✅ **Multiple Scenarios** - 6 different retrieval demonstrations  
✅ **Quality Analysis** - Compare results across k values  
✅ **Config-Driven** - Switch between ChromaDB and Pinecone  
✅ **Single File** - All code in main.py (~450 lines)  
✅ **Clear Output** - Detailed console output for each strategy  
✅ **Standard OpenAI** - No Azure required

## Retrieval Strategies Demonstrated

### 1. Basic Similarity Search

Find top-k most similar documents:

```python
results = vectorstore.similarity_search(query, k=3)
```

### 2. Similarity Search with Scores

Get relevance scores for each result:

```python
results = vectorstore.similarity_search_with_score(query, k=3)
# Returns: [(doc, 0.8542), (doc, 0.7234), ...]
```

### 3. MMR (Maximum Marginal Relevance)

Balance relevance with diversity:

```python
results = vectorstore.max_marginal_relevance_search(
    query, k=4, fetch_k=10
)
```

### 4. Retriever Interface

Use LangChain's retriever abstraction:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
results = retriever.invoke(query)
```

### 5. Metadata Filtering

Filter results by document properties:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"source": "company_policy.pdf"}
    }
)
```

### 6. Quality Analysis

Compare retrieval quality across different k values:

```python
analyze_retrieval_quality(query, k_values=[1, 2, 3, 5])
```

## Prerequisites

- Python 3.12+
- OpenAI API key

## Installation

1. Set up Python environment:

```bash
cd demo-10-rag-retrieval-pipeline
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
uv pip install -e .
```

3. Configure environment:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Configuration

Edit `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...

# Vector Database Selection (chromadb or pinecone)
VECTOR_DB=chromadb

# ChromaDB Settings (if using VECTOR_DB=chromadb)
CHROMA_DB_DIR=./chroma_db
COLLECTION_NAME=company_policies

# Pinecone Settings (if using VECTOR_DB=pinecone)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=company-policies
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Usage

Run the demonstration:

```bash
uv run python main.py
```

## What the Demo Does

### Phase 1: Ingestion

1. **Load Documents** - PDF, text files, web pages
2. **Chunk Documents** - Split into 1000-char chunks with 200-char overlap
3. **Store with Embeddings** - Generate vectors and store in vector DB

### Phase 2: Retrieval Demonstrations

#### Scenario 1: Different K Values

Compare retrieval with k=2, k=4, k=6:

```
Query: "What are the key policies?"

[k=2] Retrieved 2 documents
  [1] Documents/company_policy.pdf...
  [2] Documents/guidelines.txt...

[k=4] Retrieved 4 documents
  [1] Documents/company_policy.pdf...
  [2] Documents/guidelines.txt...
  [3] Documents/policy.txt...
  [4] https://www.python.org/...
```

#### Scenario 2: Relevance Scores

See how relevant each result is:

```
Query: "remote work guidelines"

✓ Retrieved 3 documents with scores
  [1] Score: 0.8542 | Documents/company_policy.pdf...
  [2] Score: 0.7234 | Documents/guidelines.txt...
  [3] Score: 0.6891 | Documents/policy.txt...
```

#### Scenario 3: MMR Search

Get diverse results (avoid redundancy):

```
Query: "company policies"

✓ Retrieved 4 diverse documents
  (Balances relevance with diversity)
  [1] Documents/company_policy.pdf (page 1)
  [2] Documents/company_policy.pdf (page 3)
  [3] Documents/guidelines.txt
  [4] https://www.python.org/...
```

#### Scenario 4: Retriever Interface

Use standard LangChain retriever pattern:

```
Query: "code review process"

✓ Retriever returned 4 documents
  [1] Documents/guidelines.txt...
  [2] Documents/company_policy.pdf...
```

#### Scenario 5: Quality Analysis

Compare metrics across k values:

```
Query: "What are the guidelines?"

--- k=1 ---
  Average relevance score: 0.8542
  Top result score: 0.8542

--- k=2 ---
  Average relevance score: 0.7888
  Top result score: 0.8542
  Bottom result score: 0.7234

--- k=3 ---
  Average relevance score: 0.7556
  Top result score: 0.8542
  Bottom result score: 0.6891
```

#### Scenario 6: Document Inspection

Deep dive into a retrieved document:

```
Query: "employee benefits"

======================================================================
DOCUMENT #1 DETAILS
======================================================================

[Metadata]
  source: Documents/company_policy.pdf
  page: 4

[Content]
  Length: 892 characters
  Preview: Employee Benefits  Full-time employees are eligible for
           comprehensive benefits including health insurance...
```

## Understanding the Output

### Ingestion Phase

```
======================================================================
STEP 1: LOADING DOCUMENTS
======================================================================
[1.1] Loading PDF...
  ✓ Loaded 5 page(s) from PDF
[1.2] Loading text files...
  ✓ Loaded: guidelines.txt
  ✓ Loaded: policy.txt
[1.3] Loading web page...
  ✓ Loaded web page
✓ Total documents loaded: 8

======================================================================
STEP 2: CHUNKING DOCUMENTS
======================================================================
✓ Created 127 chunks
  - Average length: 847 characters
  - Min length: 234 characters
  - Max length: 1000 characters

======================================================================
STEP 3: STORE CHUNKS WITH EMBEDDINGS
======================================================================
🔄 Processing 127 chunks...
  - Generating embeddings with OpenAI
  - Storing in CHROMADB
✓ Successfully stored 127 chunks with embeddings!
```

### Retrieval Phase

```
======================================================================
DEMONSTRATING RETRIEVAL SCENARIOS
======================================================================

[Scenario 1] Comparing Different K Values
----------------------------------------------------------------------
[Retrieval] Similarity Search (k=2)
Query: "What are the key policies?"
✓ Retrieved 2 documents
  [1] Documents/company_policy.pdf...
  [2] Documents/guidelines.txt...
```

## Key Concepts Explained

### 1. Similarity Search

Finds documents with embeddings most similar to the query embedding using cosine similarity.

**When to use**: When you want the most relevant results regardless of diversity.

### 2. Relevance Scores

Numerical scores (0-1) indicating how similar each result is to the query.

**Lower score = More similar** (in terms of distance)

### 3. K Value

Number of results to retrieve.

- **Low k (1-3)**: Only top matches (precision)
- **Medium k (4-6)**: Balance (recommended)
- **High k (10+)**: More coverage (recall)

### 4. MMR (Maximum Marginal Relevance)

Algorithm that balances relevance with diversity to avoid redundant results.

**Parameters**:

- `k`: Final number of results
- `fetch_k`: Initial candidates to consider
- `lambda_mult`: Balance factor (0=diversity, 1=relevance)

### 5. Metadata Filtering

Restrict search to documents matching specific criteria:

```python
filter = {"source": "company_policy.pdf"}
filter = {"page": 2}
filter = {"department": "engineering"}
```

### 6. Retriever Interface

LangChain abstraction for consistent retrieval across different vector stores.

**Benefits**:

- Consistent API
- Easy to swap vector databases
- Works with LCEL chains

## Code Structure

```python
# Configuration
- Load environment variables
- Initialize embeddings (OpenAI)
- Initialize vector store (ChromaDB or Pinecone)

# Step 1: load_documents()
- Load PDF, text, web

# Step 2: chunk_documents()
- Split into chunks
- Calculate statistics

# Step 3: store_chunks()
- Generate embeddings
- Store in vector database

# Step 4: Retrieval Functions
- similarity_search_basic()
- similarity_search_with_score()
- max_marginal_relevance_search()
- retriever_interface_demo()
- retriever_with_filter()
- analyze_retrieval_quality()
- display_document_details()

# Demonstration: demonstrate_retrieval_scenarios()
- Run 6 different scenarios
- Show various retrieval strategies
```

## Switching Vector Databases

### ChromaDB (Default):

```bash
# .env
VECTOR_DB=chromadb
CHROMA_DB_DIR=./chroma_db
COLLECTION_NAME=company_policies
```

### Pinecone:

```bash
# .env
VECTOR_DB=pinecone
PINECONE_API_KEY=your-api-key
PINECONE_INDEX_NAME=company-policies
```

## Common Issues

### "OPENAI_API_KEY not found"

Copy `.env.example` to `.env` and add your API key.

### "No documents loaded"

Create `Documents/` folder and add PDF or text files.

### "MMR not supported"

Some vector stores don't support MMR. This is expected and handled gracefully.

### Low relevance scores

- Try different queries
- Adjust chunk size
- Add more documents
- Check document content quality

## Learning Objectives

After running this demo, you should understand:

1. ✓ How vector similarity search works
2. ✓ What relevance scores indicate
3. ✓ How k value affects results
4. ✓ When to use MMR for diversity
5. ✓ How to use the retriever interface
6. ✓ How metadata filtering works
7. ✓ How to analyze retrieval quality
8. ✓ What factors affect retrieval performance

## Improving Retrieval Quality

### 1. Adjust Chunk Size

```python
CHUNK_SIZE = 500   # Smaller chunks (more precise)
CHUNK_SIZE = 1500  # Larger chunks (more context)
```

### 2. Tune Chunk Overlap

```python
CHUNK_OVERLAP = 100  # Less overlap
CHUNK_OVERLAP = 300  # More overlap (better context preservation)
```

### 3. Optimize K Value

```python
k = 3   # Precision-focused
k = 5   # Balanced
k = 10  # Recall-focused
```

### 4. Use MMR for Diversity

When getting redundant results, use MMR instead of similarity search.

### 5. Add Metadata

Enrich documents with metadata for filtering:

```python
doc.metadata["department"] = "engineering"
doc.metadata["date"] = "2024-01-15"
doc.metadata["category"] = "policy"
```

## Next Steps

- **Add your documents**: Put your PDF/text files in `Documents/`
- **Try different queries**: Test retrieval with your questions
- **Adjust chunk size**: See how it affects results
- **Compare k values**: Find optimal k for your use case
- **Experiment with MMR**: Test diversity vs relevance trade-off
- **Complete RAG**: Move to demo-11 for LLM-based generation

## Related Demos

- **demo-07**: Vector database CRUD operations
- **demo-08**: Loading documents from multiple sources
- **demo-09**: RAG ingestion pipeline
- **demo-11**: Complete RAG with LLM generation

## Performance Notes

**ChromaDB**:

- ✓ Fast for < 100K documents
- ✓ No external dependencies
- ✓ Perfect for development and learning

**Pinecone**:

- ✓ Fast for millions of documents
- ✓ Cloud-based, always available
- ✓ Better for production use

## References

- [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [MMR Paper](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)

## License

Educational use only.
