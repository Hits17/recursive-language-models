"""
Recursive Language Models (RLM) - Ollama Integration

A Python implementation of RLMs that works with local Ollama models.
Based on the research by Alex L. Zhang and Omar Khattab (MIT OASYS Lab).

Paper: https://arxiv.org/abs/2512.24601
Blog: https://alexzhang13.github.io/blog/2025/rlm/

Enhanced with:
- Async/parallel execution
- Streaming output
- Budget controls
- RAG integration
"""

from .rlm_core import RLM, RLMResult, rlm_completion
from .repl import REPL, REPLResult
from .ollama_client import OllamaClient, CompletionResponse

# Enhanced modules
from .async_rlm import AsyncOllamaClient, ParallelChunkProcessor
from .streaming import StreamingOllamaClient, ProgressTracker
from .budget import BudgetManager, BudgetConfig, COST_PRESETS
from .rag import RAGEnhancedRLM, SimpleVectorStore

__version__ = "0.2.0"
__all__ = [
    # Core
    "RLM",
    "RLMResult", 
    "rlm_completion",
    "REPL",
    "REPLResult",
    "OllamaClient",
    "CompletionResponse",
    # Enhanced
    "AsyncOllamaClient",
    "ParallelChunkProcessor",
    "StreamingOllamaClient",
    "ProgressTracker",
    "BudgetManager",
    "BudgetConfig",
    "COST_PRESETS",
    "RAGEnhancedRLM",
    "SimpleVectorStore",
]

