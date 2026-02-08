"""
Enhanced RLM - Simple RAG Integration

This module provides a lightweight RAG (Retrieval-Augmented Generation)
layer that can be combined with RLM for optimal context handling:

1. Pre-filter context using semantic search
2. Apply RLM for precise reasoning over filtered results
3. Best of both worlds: fast retrieval + deep understanding

No external vector database required - uses simple TF-IDF for demo.
"""

import os
import re
import math
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import Counter
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Document:
    """A document chunk for retrieval."""
    id: str
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """A search result with score."""
    document: Document
    score: float
    highlights: List[str] = None


class SimpleVectorStore:
    """
    A simple TF-IDF based vector store for document retrieval.
    
    For production, replace with:
    - ChromaDB
    - Pinecone
    - Weaviate
    - FAISS
    - Qdrant
    """

    def __init__(self):
        self.documents: List[Document] = []
        self.tfidf_vectors: Dict[str, Dict[str, float]] = {}
        self.idf: Dict[str, float] = {}
        self.vocab: set = set()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        counts = Counter(tokens)
        total = len(tokens)
        return {term: count / total for term, count in counts.items()}

    def _compute_idf(self):
        """Compute inverse document frequency for all terms."""
        n_docs = len(self.documents)
        doc_counts = Counter()
        
        for doc_id, tf_vector in self.tfidf_vectors.items():
            for term in tf_vector.keys():
                doc_counts[term] += 1
        
        self.idf = {
            term: math.log(n_docs / (count + 1)) + 1
            for term, count in doc_counts.items()
        }

    def add_documents(self, documents: List[Document]):
        """Add documents to the store."""
        for doc in documents:
            self.documents.append(doc)
            tokens = self._tokenize(doc.content)
            self.vocab.update(tokens)
            tf = self._compute_tf(tokens)
            self.tfidf_vectors[doc.id] = tf
        
        # Recompute IDF after adding documents
        self._compute_idf()
        
        # Apply IDF weights
        for doc_id, tf_vector in self.tfidf_vectors.items():
            for term in tf_vector:
                tf_vector[term] *= self.idf.get(term, 1.0)

    def add_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> int:
        """
        Add text by chunking it into documents.
        
        Returns number of chunks created.
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_content = text[start:end]
            
            chunks.append(Document(
                id=f"chunk_{chunk_id}",
                content=chunk_content,
                metadata={"start": start, "end": end}
            ))
            
            chunk_id += 1
            start += chunk_size - overlap
        
        self.add_documents(chunks)
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for documents matching the query.
        
        Uses cosine similarity between query and document TF-IDF vectors.
        """
        query_tokens = self._tokenize(query)
        query_tf = self._compute_tf(query_tokens)
        
        # Apply IDF weights to query
        query_vector = {
            term: tf * self.idf.get(term, 1.0)
            for term, tf in query_tf.items()
        }
        
        results = []
        
        for doc in self.documents:
            doc_vector = self.tfidf_vectors.get(doc.id, {})
            
            # Compute cosine similarity
            score = self._cosine_similarity(query_vector, doc_vector)
            
            if score >= min_score:
                # Find highlights (matching terms)
                doc_tokens = set(self._tokenize(doc.content))
                query_set = set(query_tokens)
                highlights = list(doc_tokens & query_set)
                
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    highlights=highlights,
                ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]

    def _cosine_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        # Dot product
        dot = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) | set(vec2))
        
        # Magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot / (mag1 * mag2)


class RAGEnhancedRLM:
    """
    RLM enhanced with RAG pre-filtering.
    
    Workflow:
    1. Index the context
    2. Retrieve relevant chunks based on query
    3. Run RLM only on relevant chunks
    
    This reduces token usage and improves accuracy for large contexts.
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        overlap: int = 200,
        top_k: int = 5,
        min_relevance: float = 0.1,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.store = SimpleVectorStore()

    def index_context(self, context: str) -> int:
        """
        Index the context for retrieval.
        
        Returns number of chunks created.
        """
        self.store = SimpleVectorStore()  # Reset
        return self.store.add_text(
            context,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

    def retrieve(self, query: str) -> List[SearchResult]:
        """Retrieve relevant chunks for a query."""
        return self.store.search(
            query,
            top_k=self.top_k,
            min_score=self.min_relevance,
        )

    def get_filtered_context(self, query: str) -> Tuple[str, List[SearchResult]]:
        """
        Get filtered context based on query relevance.
        
        Returns:
            Tuple of (filtered_context_string, search_results)
        """
        results = self.retrieve(query)
        
        if not results:
            # Fallback to full context if no relevant chunks found
            return "", results
        
        # Combine relevant chunks
        filtered_parts = []
        for i, result in enumerate(results):
            filtered_parts.append(
                f"[Chunk {i+1} (relevance: {result.score:.2f})]\n{result.document.content}"
            )
        
        filtered_context = "\n\n---\n\n".join(filtered_parts)
        
        return filtered_context, results

    def completion(
        self,
        query: str,
        context: str,
        rlm_instance=None,
    ) -> Dict[str, Any]:
        """
        Run RAG-enhanced RLM completion.
        
        Args:
            query: The user's query
            context: The full context
            rlm_instance: Optional RLM instance to use (creates one if not provided)
            
        Returns:
            Dict with response and metadata
        """
        # Step 1: Index context
        num_chunks = self.index_context(context)
        
        # Step 2: Retrieve relevant chunks
        filtered_context, results = self.get_filtered_context(query)
        
        # Step 3: Run RLM on filtered context
        if rlm_instance is None:
            from .rlm_core import RLM
            rlm_instance = RLM(verbose=True)
        
        # If we found relevant chunks, use filtered context
        if filtered_context:
            rlm_result = rlm_instance.completion(query, filtered_context)
        else:
            # Fall back to full context
            rlm_result = rlm_instance.completion(query, context)
        
        return {
            "response": rlm_result.response,
            "total_chunks": num_chunks,
            "relevant_chunks": len(results),
            "original_context_size": len(context),
            "filtered_context_size": len(filtered_context),
            "compression_ratio": len(filtered_context) / len(context) if context else 0,
            "relevance_scores": [r.score for r in results],
            "rlm_iterations": rlm_result.total_iterations,
            "tokens_used": rlm_result.total_tokens,
        }


def demo_rag():
    """Demonstrate RAG enhancement."""
    print("=" * 60)
    print("DEMO: RAG-Enhanced RLM")
    print("=" * 60)
    
    # Create a larger context with diverse topics
    context = """
    Chapter 1: Introduction to Python
    Python is a high-level programming language known for its simplicity.
    It was created by Guido van Rossum in 1991.
    Python supports multiple programming paradigms.
    
    Chapter 2: Data Structures
    Lists are ordered collections in Python.
    Dictionaries store key-value pairs.
    Sets contain unique elements.
    
    Chapter 3: Machine Learning
    Machine learning is a subset of artificial intelligence.
    Neural networks are inspired by biological brains.
    Deep learning uses multiple layers of neurons.
    
    Chapter 4: Recursion
    Recursion is when a function calls itself.
    Base cases prevent infinite recursion.
    Recursive solutions can be elegant but may use more memory.
    Tail recursion can be optimized by some compilers.
    
    Chapter 5: Web Development
    Flask and Django are popular Python web frameworks.
    REST APIs use HTTP methods like GET and POST.
    Authentication protects web resources.
    
    Chapter 6: Database Management
    SQL is used to query relational databases.
    NoSQL databases like MongoDB store documents.
    ORMs map objects to database tables.
    """
    
    print(f"\nContext size: {len(context)} characters")
    
    # Create RAG-enhanced RLM
    rag_rlm = RAGEnhancedRLM(chunk_size=500, top_k=3)
    
    # Index the context
    num_chunks = rag_rlm.index_context(context)
    print(f"Indexed into {num_chunks} chunks")
    
    # Search for relevant content
    query = "What is recursion and how does it work?"
    print(f"\nQuery: {query}")
    
    results = rag_rlm.retrieve(query)
    
    print(f"\nFound {len(results)} relevant chunks:")
    for i, result in enumerate(results):
        print(f"\n  [{i+1}] Score: {result.score:.3f}")
        print(f"      Highlights: {result.highlights[:5]}")
        print(f"      Content: {result.document.content[:100]}...")
    
    # Get filtered context
    filtered, _ = rag_rlm.get_filtered_context(query)
    print(f"\nFiltered context size: {len(filtered)} chars")
    print(f"Compression: {len(filtered)/len(context)*100:.1f}% of original")


if __name__ == "__main__":
    demo_rag()
