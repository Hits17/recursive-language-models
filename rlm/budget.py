"""
Enhanced RLM - Token Budget & Cost Controls

This module provides budget management for RLM operations:
- Token limits per query
- Cost estimation and caps
- Iteration limits
- Early termination when budget exceeded

Essential for production deployments where cost control matters.
"""

import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class BudgetExceededError(Exception):
    """Raised when token or cost budget is exceeded."""
    pass


class TerminationReason(Enum):
    """Reason for RLM termination."""
    ANSWER_FOUND = "answer_found"
    MAX_ITERATIONS = "max_iterations"
    TOKEN_BUDGET = "token_budget"
    COST_BUDGET = "cost_budget"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class BudgetConfig:
    """Configuration for RLM budget limits."""
    
    # Token limits
    max_input_tokens: int = 100000  # Max tokens in input context
    max_output_tokens_per_call: int = 4096  # Max tokens per LLM call
    max_total_tokens: int = 500000  # Max total tokens across all calls
    
    # Cost limits (for cloud APIs)
    max_cost_per_query: float = 1.0  # Max cost in dollars
    input_token_cost: float = 0.0  # Cost per 1K input tokens (0 for local)
    output_token_cost: float = 0.0  # Cost per 1K output tokens (0 for local)
    
    # Iteration limits
    max_iterations: int = 50
    max_code_executions: int = 100
    
    # Time limits
    max_query_time: float = 300.0  # 5 minutes max
    max_llm_call_time: float = 60.0  # 1 minute per LLM call
    
    # Warning thresholds (percentage)
    warn_at_percentage: float = 0.8  # Warn when 80% of budget used


@dataclass
class BudgetUsage:
    """Track current budget usage."""
    tokens_used: int = 0
    cost_incurred: float = 0.0
    iterations: int = 0
    code_executions: int = 0
    elapsed_time: float = 0.0
    
    # Breakdown
    input_tokens: int = 0
    output_tokens: int = 0


class BudgetManager:
    """
    Manages token and cost budgets for RLM operations.
    
    Usage:
        budget = BudgetManager(BudgetConfig(max_total_tokens=100000))
        
        # Before each LLM call
        budget.check_can_proceed()
        
        # After each LLM call
        budget.record_usage(input_tokens=1000, output_tokens=500)
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self.usage = BudgetUsage()
        self.start_time = time.time()
        self.warnings_issued: Dict[str, bool] = {}

    def reset(self):
        """Reset usage for a new query."""
        self.usage = BudgetUsage()
        self.start_time = time.time()
        self.warnings_issued = {}

    def record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        is_iteration: bool = False,
        is_code_execution: bool = False,
    ):
        """Record token and operation usage."""
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.tokens_used = self.usage.input_tokens + self.usage.output_tokens
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * self.config.input_token_cost
        output_cost = (output_tokens / 1000) * self.config.output_token_cost
        self.usage.cost_incurred += input_cost + output_cost
        
        if is_iteration:
            self.usage.iterations += 1
        
        if is_code_execution:
            self.usage.code_executions += 1
        
        self.usage.elapsed_time = time.time() - self.start_time
        
        # Check for warnings
        self._check_warnings()

    def _check_warnings(self):
        """Issue warnings when approaching limits."""
        warn_pct = self.config.warn_at_percentage
        
        # Token warning
        if not self.warnings_issued.get("tokens"):
            if self.usage.tokens_used > self.config.max_total_tokens * warn_pct:
                print(f"⚠️  Warning: {self.usage.tokens_used:,} tokens used "
                      f"({self.usage.tokens_used / self.config.max_total_tokens * 100:.0f}% of budget)")
                self.warnings_issued["tokens"] = True
        
        # Cost warning
        if not self.warnings_issued.get("cost"):
            if self.usage.cost_incurred > self.config.max_cost_per_query * warn_pct:
                print(f"⚠️  Warning: ${self.usage.cost_incurred:.4f} spent "
                      f"({self.usage.cost_incurred / self.config.max_cost_per_query * 100:.0f}% of budget)")
                self.warnings_issued["cost"] = True
        
        # Time warning
        if not self.warnings_issued.get("time"):
            if self.usage.elapsed_time > self.config.max_query_time * warn_pct:
                print(f"⚠️  Warning: {self.usage.elapsed_time:.0f}s elapsed "
                      f"({self.usage.elapsed_time / self.config.max_query_time * 100:.0f}% of time limit)")
                self.warnings_issued["time"] = True

    def check_can_proceed(self) -> TerminationReason | None:
        """
        Check if we can proceed with another operation.
        
        Returns:
            None if OK to proceed, or TerminationReason if should stop
        """
        # Check token budget
        if self.usage.tokens_used >= self.config.max_total_tokens:
            return TerminationReason.TOKEN_BUDGET
        
        # Check cost budget
        if self.usage.cost_incurred >= self.config.max_cost_per_query:
            return TerminationReason.COST_BUDGET
        
        # Check iteration limit
        if self.usage.iterations >= self.config.max_iterations:
            return TerminationReason.MAX_ITERATIONS
        
        # Check time limit
        self.usage.elapsed_time = time.time() - self.start_time
        if self.usage.elapsed_time >= self.config.max_query_time:
            return TerminationReason.TIMEOUT
        
        return None

    def get_remaining(self) -> Dict[str, Any]:
        """Get remaining budget."""
        return {
            "tokens": self.config.max_total_tokens - self.usage.tokens_used,
            "cost": self.config.max_cost_per_query - self.usage.cost_incurred,
            "iterations": self.config.max_iterations - self.usage.iterations,
            "time": self.config.max_query_time - self.usage.elapsed_time,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        return {
            "tokens_used": self.usage.tokens_used,
            "tokens_limit": self.config.max_total_tokens,
            "tokens_percentage": self.usage.tokens_used / self.config.max_total_tokens * 100,
            "cost_incurred": self.usage.cost_incurred,
            "cost_limit": self.config.max_cost_per_query,
            "iterations": self.usage.iterations,
            "code_executions": self.usage.code_executions,
            "elapsed_time": self.usage.elapsed_time,
        }

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        s = self.get_summary()
        return f"""
Budget Summary:
  Tokens: {s['tokens_used']:,} / {s['tokens_limit']:,} ({s['tokens_percentage']:.1f}%)
  Cost: ${s['cost_incurred']:.4f} / ${s['cost_limit']:.2f}
  Iterations: {s['iterations']} / {self.config.max_iterations}
  Time: {s['elapsed_time']:.1f}s / {self.config.max_query_time:.0f}s
"""


# ============== Cost Presets for Different Providers ==============

COST_PRESETS = {
    "ollama_local": BudgetConfig(
        input_token_cost=0.0,
        output_token_cost=0.0,
        max_cost_per_query=0.0,  # Free!
    ),
    "gpt4": BudgetConfig(
        input_token_cost=0.03,  # $30 per 1M tokens
        output_token_cost=0.06,
        max_cost_per_query=5.0,
    ),
    "gpt4_mini": BudgetConfig(
        input_token_cost=0.00015,
        output_token_cost=0.0006,
        max_cost_per_query=1.0,
    ),
    "claude3_sonnet": BudgetConfig(
        input_token_cost=0.003,
        output_token_cost=0.015,
        max_cost_per_query=2.0,
    ),
    "qwen_cloud": BudgetConfig(
        input_token_cost=0.0005,
        output_token_cost=0.001,
        max_cost_per_query=0.5,
    ),
}


def demo_budget_manager():
    """Demonstrate budget management."""
    print("=" * 60)
    print("DEMO: Budget Management")
    print("=" * 60)
    
    # Create a tight budget for demo
    config = BudgetConfig(
        max_total_tokens=5000,
        max_iterations=5,
        max_query_time=30.0,
    )
    
    budget = BudgetManager(config)
    
    print(f"\nStarting with budget:")
    print(f"  Max tokens: {config.max_total_tokens:,}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Max time: {config.max_query_time}s")
    
    # Simulate some operations
    for i in range(10):
        # Check if we can proceed
        reason = budget.check_can_proceed()
        if reason:
            print(f"\n🛑 Stopping: {reason.value}")
            break
        
        # Simulate an LLM call
        budget.record_usage(
            input_tokens=800,
            output_tokens=400,
            is_iteration=True,
        )
        
        print(f"\nIteration {i+1}:")
        remaining = budget.get_remaining()
        print(f"  Tokens remaining: {remaining['tokens']:,}")
        print(f"  Iterations remaining: {remaining['iterations']}")
    
    print(budget.format_summary())


if __name__ == "__main__":
    demo_budget_manager()
