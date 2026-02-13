"""
Demo 10: RAG Retrieval Pipeline

This demo focuses on the RETRIEVAL phase of RAG:
1. Load documents from multiple sources
2. Chunk documents
3. Generate embeddings and store in vector database
4. Demonstrate various retrieval strategies and scenarios

Focus: Understanding retrieval quality, not answer generation
(For complete RAG with generation, see demo-11)

Supports two vector databases via configuration:
- ChromaDB (local, file-based)
- Pinecone (cloud-based)

Usage:
    # Set VECTOR_DB=chromadb or pinecone in .env
    uv run python main.py
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
VECTOR_DB = os.getenv("VECTOR_DB", "chromadb").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Document source configuration
DOCS_DIR = Path("Documents")
PDF_FILE = DOCS_DIR / "company_policy.pdf"
WEB_URL = "https://www.python.org/"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

print("=" * 70)
print("RAG RETRIEVAL PIPELINE CONFIGURATION")
print("=" * 70)
print(f"Vector Database: {VECTOR_DB.upper()}")
print(f"Chunk Size: {CHUNK_SIZE} characters")
print(f"Chunk Overlap: {CHUNK_OVERLAP} characters")
print("=" * 70)

# ============================================================================
# INITIALIZE EMBEDDINGS
# ============================================================================
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)
print("\n✓ OpenAI embeddings initialized: text-embedding-3-small")

# ============================================================================
# INITIALIZE VECTOR STORE (Config-Driven)
# ============================================================================
vectorstore = None

if VECTOR_DB == "chromadb":
    from langchain_chroma import Chroma
    
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company_policies")
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    
    print(f"✓ ChromaDB initialized")
    print(f"  - Storage: {CHROMA_DB_DIR}")
    print(f"  - Collection: {COLLECTION_NAME}")

elif VECTOR_DB == "pinecone":
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-policies")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables (required for Pinecone)")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        print(f"✓ Created Pinecone index: {PINECONE_INDEX_NAME}")
    
    # Initialize vector store
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    
    print(f"✓ Pinecone initialized")
    print(f"  - Index: {PINECONE_INDEX_NAME}")
    print(f"  - Cloud: {PINECONE_CLOUD}/{PINECONE_REGION}")

else:
    raise ValueError(f"Unsupported VECTOR_DB: {VECTOR_DB}. Use 'chromadb' or 'pinecone'")

print(f"✓ Vector store ready!\n")


# ============================================================================
# STEP 1: LOAD DOCUMENTS
# ============================================================================
def load_documents() -> List[Document]:
    """Load documents from PDF, text files, and web."""
    print("=" * 70)
    print("STEP 1: LOADING DOCUMENTS")
    print("=" * 70)
    
    all_docs = []
    
    # Load PDF
    print("\n[1.1] Loading PDF...")
    if PDF_FILE.exists():
        try:
            pdf_loader = PyPDFLoader(str(PDF_FILE))
            pdf_docs = pdf_loader.load()
            all_docs.extend(pdf_docs)
            print(f"  ✓ Loaded {len(pdf_docs)} page(s) from PDF")
        except Exception as e:
            print(f"  ✗ Error loading PDF: {e}")
    else:
        print(f"  ⚠️  PDF not found: {PDF_FILE}")
    
    # Load text files
    print("\n[1.2] Loading text files...")
    txt_files = list(DOCS_DIR.glob("*.txt"))
    if txt_files:
        for txt_file in txt_files:
            try:
                text_loader = TextLoader(str(txt_file), encoding="utf-8")
                text_docs = text_loader.load()
                all_docs.extend(text_docs)
                print(f"  ✓ Loaded: {txt_file.name}")
            except Exception as e:
                print(f"  ✗ Error loading {txt_file.name}: {e}")
    else:
        print("  ⚠️  No .txt files found")
    
    # Load web page
    print("\n[1.3] Loading web page...")
    try:
        web_loader = WebBaseLoader(WEB_URL)
        web_docs = web_loader.load()
        all_docs.extend(web_docs)
        print(f"  ✓ Loaded web page")
        print(f"  ✓ Content length: {len(web_docs[0].page_content):,} characters")
    except Exception as e:
        print(f"  ✗ Error loading web page: {e}")
    
    print(f"\n✓ Total documents loaded: {len(all_docs)}")
    return all_docs


# ============================================================================
# STEP 2: CHUNK DOCUMENTS
# ============================================================================
def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
    print("\n" + "=" * 70)
    print("STEP 2: CHUNKING DOCUMENTS")
    print("=" * 70)
    
    if not documents:
        print("⚠️  No documents to chunk!")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Calculate statistics
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    
    print(f"\n✓ Created {len(chunks)} chunks")
    print(f"  - Average length: {avg_length:.0f} characters")
    print(f"  - Min length: {min(chunk_lengths) if chunk_lengths else 0} characters")
    print(f"  - Max length: {max(chunk_lengths) if chunk_lengths else 0} characters")
    
    return chunks


# ============================================================================
# STEP 3: STORE CHUNKS IN VECTOR DATABASE
# ============================================================================
def store_chunks(chunks: List[Document]) -> None:
    """Generate embeddings and store chunks in vector database."""
    print("\n" + "=" * 70)
    print("STEP 3: STORE CHUNKS WITH EMBEDDINGS")
    print("=" * 70)
    
    if not chunks:
        print("⚠️  No chunks to store!")
        return
    
    print(f"\n🔄 Processing {len(chunks)} chunks...")
    print("  - Generating embeddings with OpenAI")
    print(f"  - Storing in {VECTOR_DB.upper()}")
    
    try:
        vectorstore.add_documents(chunks)
        print(f"\n✓ Successfully stored {len(chunks)} chunks with embeddings!")
    except Exception as e:
        print(f"\n✗ Error storing chunks: {e}")
        raise


# ============================================================================
# STEP 4: RETRIEVAL STRATEGIES
# ============================================================================

def similarity_search_basic(query: str, k: int = 3) -> List[Document]:
    """Basic similarity search with k results."""
    print(f"\n[Retrieval] Similarity Search (k={k})")
    print(f"Query: \"{query}\"")
    
    results = vectorstore.similarity_search(query, k=k)
    
    print(f"✓ Retrieved {len(results)} documents")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.metadata.get('source', 'Unknown')[:50]}...")
    
    return results


def similarity_search_with_score(query: str, k: int = 3) -> List[tuple]:
    """Similarity search with relevance scores."""
    print(f"\n[Retrieval] Similarity Search with Scores (k={k})")
    print(f"Query: \"{query}\"")
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    print(f"✓ Retrieved {len(results)} documents with scores")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  [{i}] Score: {score:.4f} | {doc.metadata.get('source', 'Unknown')[:40]}...")
    
    return results


def max_marginal_relevance_search(query: str, k: int = 3, fetch_k: int = 10) -> List[Document]:
    """
    MMR search - balances relevance with diversity.
    Fetches fetch_k candidates, returns k diverse results.
    """
    print(f"\n[Retrieval] MMR Search (k={k}, fetch_k={fetch_k})")
    print(f"Query: \"{query}\"")
    print("  (Maximizes diversity while maintaining relevance)")
    
    try:
        results = vectorstore.max_marginal_relevance_search(
            query, 
            k=k, 
            fetch_k=fetch_k
        )
        
        print(f"✓ Retrieved {len(results)} diverse documents")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.metadata.get('source', 'Unknown')[:50]}...")
        
        return results
    except Exception as e:
        print(f"✗ MMR not supported: {e}")
        return []


def retriever_interface_demo(query: str):
    """Demonstrate using retriever interface with configuration."""
    print(f"\n[Retrieval] Using Retriever Interface")
    print(f"Query: \"{query}\"")
    
    # Create retriever with specific configuration
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    results = retriever.invoke(query)
    
    print(f"✓ Retriever returned {len(results)} documents")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.metadata.get('source', 'Unknown')[:50]}...")
    
    return results


def retriever_with_filter(query: str, metadata_filter: Dict[str, Any] = None):
    """Demonstrate retrieval with metadata filtering."""
    print(f"\n[Retrieval] Retrieval with Metadata Filter")
    print(f"Query: \"{query}\"")
    if metadata_filter:
        print(f"Filter: {metadata_filter}")
    
    try:
        if VECTOR_DB == "chromadb" and metadata_filter:
            # ChromaDB uses where clause
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 3,
                    "filter": metadata_filter
                }
            )
        else:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
        
        results = retriever.invoke(query)
        
        print(f"✓ Retrieved {len(results)} filtered documents")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.metadata.get('source', 'Unknown')[:50]}...")
        
        return results
    except Exception as e:
        print(f"✗ Filtering error: {e}")
        return []


# ============================================================================
# RETRIEVAL QUALITY ANALYSIS
# ============================================================================

def analyze_retrieval_quality(query: str, k_values: List[int] = [1, 2, 3, 5]):
    """Compare retrieval quality across different k values."""
    print("\n" + "=" * 70)
    print("RETRIEVAL QUALITY ANALYSIS")
    print("=" * 70)
    print(f"\nQuery: \"{query}\"")
    print(f"Testing k values: {k_values}")
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        if results:
            avg_score = sum(score for _, score in results) / len(results)
            print(f"  Average relevance score: {avg_score:.4f}")
            print(f"  Top result score: {results[0][1]:.4f}")
            print(f"  Bottom result score: {results[-1][1]:.4f}")
        else:
            print("  No results found")


def display_document_details(doc: Document, index: int = 1):
    """Display detailed information about a retrieved document."""
    print(f"\n{'=' * 70}")
    print(f"DOCUMENT #{index} DETAILS")
    print(f"{'=' * 70}")
    
    # Metadata
    print("\n[Metadata]")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
    
    # Content
    print("\n[Content]")
    print(f"  Length: {len(doc.page_content)} characters")
    print(f"  Preview (first 300 chars):")
    print(f"  {doc.page_content[:300].replace(chr(10), ' ')}...")
    
    print(f"{'=' * 70}")


# ============================================================================
# DEMONSTRATION SCENARIOS
# ============================================================================

def demonstrate_retrieval_scenarios():
    """Run comprehensive retrieval demonstrations."""
    print("\n" + "=" * 70)
    print("DEMONSTRATING RETRIEVAL SCENARIOS")
    print("=" * 70)
    
    # Scenario 1: Different k values
    print("\n[Scenario 1] Comparing Different K Values")
    print("-" * 70)
    
    query1 = "What are the key policies?"
    
    for k in [2, 4, 6]:
        similarity_search_basic(query1, k=k)
    
    # Scenario 2: Retrieval with scores
    print("\n\n[Scenario 2] Retrieval with Relevance Scores")
    print("-" * 70)
    
    query2 = "remote work guidelines"
    similarity_search_with_score(query2, k=3)
    
    # Scenario 3: MMR for diversity
    print("\n\n[Scenario 3] MMR Search for Diverse Results")
    print("-" * 70)
    
    query3 = "company policies"
    max_marginal_relevance_search(query3, k=4, fetch_k=10)
    
    # Scenario 4: Retriever interface
    print("\n\n[Scenario 4] Using Retriever Interface")
    print("-" * 70)
    
    query4 = "code review process"
    retriever_interface_demo(query4)
    
    # Scenario 5: Quality analysis
    print("\n\n[Scenario 5] Retrieval Quality Analysis")
    print("-" * 70)
    
    query5 = "What are the guidelines?"
    analyze_retrieval_quality(query5, k_values=[1, 2, 3, 5])
    
    # Scenario 6: Detailed document inspection
    print("\n\n[Scenario 6] Detailed Document Inspection")
    print("-" * 70)
    
    query6 = "employee benefits"
    results = similarity_search_with_score(query6, k=2)
    
    if results:
        print("\nInspecting top result in detail:")
        display_document_details(results[0][0], index=1)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run the complete demonstration."""
    print("\n" + "=" * 70)
    print("DEMO 10: RAG RETRIEVAL PIPELINE")
    print("=" * 70)
    
    # Steps 1-3: Ingestion (Load, Chunk, Store)
    documents = load_documents()
    
    if not documents:
        print("\n⚠️  No documents loaded. Please add documents to the Documents/ folder.")
        return
    
    chunks = chunk_documents(documents)
    
    if not chunks:
        print("\n⚠️  No chunks created.")
        return
    
    store_chunks(chunks)
    
    # Step 4: Demonstrate various retrieval scenarios
    demonstrate_retrieval_scenarios()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ RAG RETRIEVAL PIPELINE DEMONSTRATION FINISHED!")
    print("=" * 70)
    print("\n📋 Summary:")
    print(f"  1. Loaded {len(documents)} documents")
    print(f"  2. Created {len(chunks)} chunks")
    print(f"  3. Stored chunks with embeddings in {VECTOR_DB.upper()}")
    print(f"  4. Demonstrated 6 retrieval scenarios")
    print("\n🎯 Retrieval Strategies Demonstrated:")
    print("  • Basic similarity search (different k values)")
    print("  • Similarity search with relevance scores")
    print("  • MMR search for diverse results")
    print("  • Retriever interface configuration")
    print("  • Retrieval quality analysis")
    print("  • Detailed document inspection")
    print("\n💡 Next Steps:")
    print("  • For complete RAG with LLM generation, see demo-11")
    print("  • Experiment with different queries")
    print("  • Try different chunk sizes")
    print("  • Add your own documents")
    print("=" * 70)


if __name__ == "__main__":
    main()
