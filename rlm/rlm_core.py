"""
Recursive Language Model (RLM) Core Implementation

This module provides the main RLM class that wraps language models
to enable recursive self-calling through a REPL environment.

Based on: "Recursive Language Models" by Zhang, Kraska, and Khattab
Paper: https://arxiv.org/abs/2512.24601
"""

import os
import re
import json
import time
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime

from .repl import REPL, REPLResult
from .ollama_client import OllamaClient, CompletionResponse


@dataclass
class RLMResult:
    """Result from an RLM completion."""
    response: str
    success: bool
    total_iterations: int
    total_tokens: int
    cost_estimate: float
    execution_time: float
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


# ============== System Prompts ==============

SYSTEM_PROMPT = """You are an AI assistant with access to a Python REPL environment for processing data. 
The user's input context is stored in a variable called `context` in your REPL environment.

## Environment Capabilities
You have access to a Python REPL where you can:
1. **Examine the context**: Use `peek(start, length)`, `head(n)`, `tail(n)` to view portions
2. **Search**: Use `grep(pattern)` to find matching lines
3. **Process**: Write Python code to analyze and transform data
4. **Chunk**: Use `chunk(size, overlap)` to split large contexts
5. **Sub-query**: Use `ask_llm(prompt, context_subset)` to recursively query an LLM

## Available Helper Functions
- `peek(start=0, length=2000)`: View a portion of the context
- `head(n=10)`: Get first n lines
- `tail(n=10)`: Get last n lines  
- `grep(pattern, ignore_case=True)`: Find lines matching regex pattern
- `chunk(chunk_size=10000, overlap=200)`: Split context into chunks
- `count_lines()`: Count total lines
- `ask_llm(prompt, context_subset=None)`: Ask a sub-LLM a question

## Response Format
You can interact with the REPL by writing code in ```python blocks.
When you have the final answer, respond with FINAL(your_answer_here) or FINAL_VAR(variable_name).

## Important Guidelines
1. Start by exploring the context structure (peek, head, count_lines)
2. Use grep and pattern matching to narrow down relevant information
3. For complex tasks, break them into smaller sub-queries using ask_llm()
4. Be systematic and methodical in your analysis
5. If the context is very large, chunk it and process pieces

## Example Interaction
```python
# First, understand the context
print(f"Context length: {len(context)} characters")
print(f"Lines: {count_lines()}")
print("First 500 chars:")
print(peek(0, 500))
```

Then based on what you learn, continue processing until you can provide FINAL(answer)."""

CODE_BLOCK_PATTERN = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)
FINAL_PATTERN = re.compile(r'FINAL\((.*?)\)', re.DOTALL)
FINAL_VAR_PATTERN = re.compile(r'FINAL_VAR\((\w+)\)', re.DOTALL)


class RLM:
    """
    Recursive Language Model.
    
    A wrapper around language models that enables recursive self-calling
    through a REPL environment. This allows handling of unbounded context
    lengths by letting the LLM programmatically interact with its input.
    
    Usage:
        rlm = RLM(model_name="qwen3:latest")
        result = rlm.completion(
            query="Find all user IDs with 'entity' labeled questions",
            context=huge_context_string
        )
        print(result.response)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        max_iterations: int = 50,
        max_recursion_depth: int = 1,
        temperature: float = 0.7,
        verbose: bool = True,
        log_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        restricted_repl: bool = True,
        **kwargs
    ):
        """
        Initialize the RLM.

        Args:
            model_name: Ollama model to use (default: from env or 'qwen3:latest')
            host: Ollama host URL (default: from env or 'http://localhost:11434')
            max_iterations: Maximum REPL iterations before forcing termination
            max_recursion_depth: Maximum depth of recursive LLM calls
            temperature: Sampling temperature
            verbose: Whether to print execution progress
            log_dir: Directory to save execution logs
            system_prompt: Custom system prompt (uses default if None)
            restricted_repl: Whether to apply security restrictions in REPL
        """
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen3:latest")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.max_iterations = int(os.getenv("RLM_MAX_ITERATIONS", max_iterations))
        self.max_recursion_depth = int(os.getenv("RLM_MAX_RECURSION_DEPTH", max_recursion_depth))
        self.temperature = temperature
        self.verbose = verbose or os.getenv("RLM_VERBOSE", "false").lower() == "true"
        self.log_dir = log_dir or os.getenv("RLM_LOG_DIR", "./logs")
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.restricted_repl = restricted_repl
        self.extra_kwargs = kwargs

        # Initialize the Ollama client
        self.client = OllamaClient(
            model_name=self.model_name,
            host=self.host,
            temperature=self.temperature,
            **kwargs
        )

        # Create log directory if needed
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        if self.verbose:
            self._log(f"RLM initialized with model: {self.model_name}")

    def _log(self, message: str, level: str = "INFO") -> None:
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def _create_sub_llm_fn(self, depth: int) -> Callable[[str], str]:
        """
        Create a function for sub-LLM calls within the REPL.
        
        This enables the recursive nature of RLMs by allowing the root LLM
        to spawn sub-queries.
        """
        if depth >= self.max_recursion_depth:
            def no_recursion(prompt: str) -> str:
                return "[Maximum recursion depth reached. Cannot spawn sub-LLM.]"
            return no_recursion

        def sub_llm_call(prompt: str) -> str:
            """Make a sub-LLM call."""
            self._log(f"Sub-LLM call at depth {depth + 1}: {prompt[:100]}...")
            
            try:
                response = self.client.completion(prompt)
                return response.content
            except Exception as e:
                return f"[Error in sub-LLM call: {str(e)}]"

        return sub_llm_call

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from the LLM response."""
        matches = CODE_BLOCK_PATTERN.findall(text)
        return matches

    def _check_for_final(self, text: str, repl: REPL) -> Optional[str]:
        """Check if the response contains a final answer."""
        # Check for FINAL_VAR(variable_name)
        var_match = FINAL_VAR_PATTERN.search(text)
        if var_match:
            var_name = var_match.group(1).strip()
            value = repl.get_variable(var_name)
            if value is not None:
                return str(value)
            else:
                return f"[Error: Variable '{var_name}' not found]"

        # Check for FINAL(answer)
        final_match = FINAL_PATTERN.search(text)
        if final_match:
            return final_match.group(1).strip()

        return None

    def completion(
        self,
        query: str,
        context: str = "",
        **kwargs
    ) -> RLMResult:
        """
        Generate a completion using the RLM.

        This is the main entry point, designed to be a drop-in replacement
        for standard LLM completion calls.

        Args:
            query: The user's query/question
            context: The input context to process (can be arbitrarily long)
            **kwargs: Additional parameters

        Returns:
            RLMResult with the response and metadata
        """
        start_time = time.time()
        total_tokens = 0
        trajectory = []

        self._log(f"Starting RLM completion")
        self._log(f"Query: {query[:100]}...")
        self._log(f"Context length: {len(context)} characters")

        # Initialize REPL with context
        repl = REPL(
            context=context,
            ask_llm_fn=self._create_sub_llm_fn(depth=0),
            restricted=self.restricted_repl,
        )

        # Build initial messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_initial_prompt(query, context)},
        ]

        final_answer = None
        error = None

        for iteration in range(self.max_iterations):
            self._log(f"Iteration {iteration + 1}/{self.max_iterations}")

            try:
                # Get LLM response
                response = self.client.completion(messages)
                total_tokens += response.total_tokens
                assistant_message = response.content

                self._log(f"LLM response ({len(assistant_message)} chars)")

                # Record trajectory
                trajectory.append({
                    "iteration": iteration + 1,
                    "type": "llm_response",
                    "content": assistant_message,
                    "tokens": response.total_tokens,
                })

                # Check for final answer
                final_answer = self._check_for_final(assistant_message, repl)
                if final_answer:
                    self._log(f"Final answer found: {final_answer[:100]}...")
                    break

                # Extract and execute code blocks
                code_blocks = self._extract_code_blocks(assistant_message)
                
                if not code_blocks:
                    # No code to execute, add message and continue
                    messages.append({"role": "assistant", "content": assistant_message})
                    messages.append({
                        "role": "user",
                        "content": "Please continue. Write Python code to explore the context, or provide FINAL(answer) when you have the answer."
                    })
                    continue

                # Execute each code block
                repl_outputs = []
                for code in code_blocks:
                    self._log(f"Executing code block ({len(code)} chars)")
                    result = repl.execute(code)
                    
                    trajectory.append({
                        "iteration": iteration + 1,
                        "type": "code_execution",
                        "code": code,
                        "output": result.output,
                        "success": result.success,
                        "error": result.error,
                    })

                    if result.success:
                        repl_outputs.append(f"[Output]:\n{result.output}")
                    else:
                        repl_outputs.append(f"[Error]: {result.error}")

                # Add to conversation
                messages.append({"role": "assistant", "content": assistant_message})
                
                output_summary = "\n\n".join(repl_outputs)
                messages.append({
                    "role": "user",
                    "content": f"Code execution results:\n{output_summary}\n\nContinue analysis or provide FINAL(answer) when done."
                })

            except Exception as e:
                error = str(e)
                self._log(f"Error: {error}", level="ERROR")
                trajectory.append({
                    "iteration": iteration + 1,
                    "type": "error",
                    "error": error,
                })
                break

        execution_time = time.time() - start_time

        # If no final answer was found, try to extract from last response
        if final_answer is None and messages:
            last_assistant = next(
                (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
                None
            )
            if last_assistant:
                # Try to find any answer-like content
                final_answer = last_assistant
                self._log("No explicit FINAL() found, using last response")

        # Save log if directory specified
        if self.log_dir:
            self._save_log(query, context, trajectory, final_answer, execution_time)

        return RLMResult(
            response=final_answer or "[No answer generated]",
            success=final_answer is not None and error is None,
            total_iterations=len([t for t in trajectory if t["type"] == "llm_response"]),
            total_tokens=total_tokens,
            cost_estimate=self._estimate_cost(total_tokens),
            execution_time=execution_time,
            trajectory=trajectory,
            error=error,
        )

    def _build_initial_prompt(self, query: str, context: str) -> str:
        """Build the initial user prompt."""
        context_info = f"Context length: {len(context)} characters ({len(context.split(chr(10)))} lines)"
        
        return f"""I have a query to answer. The context is stored in the `context` variable in your REPL environment.

## Context Information
{context_info}

## Query
{query}

Please use the REPL environment to explore and analyze the context, then provide your final answer using FINAL(answer) or FINAL_VAR(variable_name)."""

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate the cost (returns 0 for local Ollama models)."""
        # Local Ollama has no API cost
        return 0.0

    def _save_log(
        self,
        query: str,
        context: str,
        trajectory: List[Dict],
        answer: Optional[str],
        execution_time: float
    ) -> None:
        """Save execution log to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"rlm_{timestamp}.jsonl")

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "query": query,
            "context_length": len(context),
            "answer": answer,
            "execution_time": execution_time,
            "trajectory": trajectory,
        }

        try:
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2)
            self._log(f"Log saved to: {log_file}")
        except Exception as e:
            self._log(f"Failed to save log: {e}", level="WARN")


# ============== Convenience Functions ==============

def rlm_completion(
    query: str,
    context: str = "",
    model_name: Optional[str] = None,
    verbose: bool = True,
    **kwargs
) -> str:
    """
    Convenience function for quick RLM completions.

    Args:
        query: The query to answer
        context: The context to process
        model_name: Model to use (default: qwen3:latest)
        verbose: Whether to print progress
        **kwargs: Additional RLM parameters

    Returns:
        The answer as a string
    """
    rlm = RLM(model_name=model_name, verbose=verbose, **kwargs)
    result = rlm.completion(query, context)
    return result.response


if __name__ == "__main__":
    # Quick test
    print("Testing RLM Core...")
    
    test_context = """
Entry 1: Date: Jan 01, 2024 || User: 12345 || Item: Apple
Entry 2: Date: Jan 02, 2024 || User: 67890 || Item: Banana
Entry 3: Date: Jan 03, 2024 || User: 12345 || Item: Cherry
Entry 4: Date: Jan 04, 2024 || User: 11111 || Item: Date
Entry 5: Date: Jan 05, 2024 || User: 67890 || Item: Elderberry
"""
    
    rlm = RLM(verbose=True)
    result = rlm.completion(
        query="Count how many entries are associated with user 12345",
        context=test_context
    )
    
    print(f"\n{'='*50}")
    print(f"Final Answer: {result.response}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.total_iterations}")
    print(f"Tokens used: {result.total_tokens}")
    print(f"Time: {result.execution_time:.2f}s")
