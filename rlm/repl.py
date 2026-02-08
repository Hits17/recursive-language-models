"""
REPL Environment for RLM

A Python REPL (Read-Eval-Print Loop) environment that allows the LLM
to execute Python code and interact with the context stored in memory.
"""

import sys
import io
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, List
from contextlib import redirect_stdout, redirect_stderr


@dataclass
class REPLResult:
    """Result from executing code in the REPL."""
    success: bool
    output: str
    error: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


class REPL:
    """
    A Python REPL environment for RLM.
    
    This REPL provides a sandboxed execution environment where the LLM
    can execute Python code to interact with the context. The context
    is stored as a variable that can be manipulated programmatically.
    
    Key features:
    - Safe execution with timeout and output capture
    - Pre-loaded context as 'context' variable
    - Sub-LLM call function available as 'ask_llm()'
    - Basic security restrictions on dangerous operations
    """

    # Dangerous modules/functions to restrict
    RESTRICTED_NAMES = {
        "__import__", "eval", "exec", "compile",
        "open", "input", "breakpoint",
        "exit", "quit", "help",
    }

    RESTRICTED_MODULES = {
        "os", "sys", "subprocess", "shutil",
        "socket", "http", "urllib", "ftplib",
        "importlib", "pickle", "marshal",
    }

    def __init__(
        self,
        context: str = "",
        ask_llm_fn: Optional[Callable[[str], str]] = None,
        max_output_length: int = 50000,
        restricted: bool = True,
        allowed_imports: Optional[List[str]] = None,
    ):
        """
        Initialize the REPL environment.

        Args:
            context: The input context to be processed (stored as 'context' variable)
            ask_llm_fn: Function to call sub-LLM (signature: (prompt) -> response)
            max_output_length: Maximum length of output before truncation
            restricted: Whether to apply security restrictions
            allowed_imports: List of modules allowed for import (if restricted=True)
        """
        self.context = context
        self.ask_llm_fn = ask_llm_fn
        self.max_output_length = max_output_length
        self.restricted = restricted
        self.allowed_imports = allowed_imports or ["re", "json", "math", "random", "collections", "itertools", "functools"]
        
        # Initialize the namespace with context and helper functions
        self.namespace: Dict[str, Any] = self._create_namespace()
        
        # Execution history
        self.history: List[Dict[str, Any]] = []

    def _create_namespace(self) -> Dict[str, Any]:
        """Create the initial namespace with context and helper functions."""
        namespace = {
            # The context is available as a variable
            "context": self.context,
            
            # Useful builtins
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "reversed": reversed,
            "any": any,
            "all": all,
            "isinstance": isinstance,
            "type": type,
            "print": print,  # Will be captured
            "repr": repr,
            
            # Common imports pre-loaded
            "re": __import__("re"),
            "json": __import__("json"),
            "math": __import__("math"),
            "random": __import__("random"),
            
            # Helper functions
            "peek": self._peek,
            "chunk": self._chunk,
            "grep": self._grep,
            "count_lines": self._count_lines,
            "head": self._head,
            "tail": self._tail,
        }
        
        # Add sub-LLM call function if provided
        if self.ask_llm_fn:
            namespace["ask_llm"] = self._safe_ask_llm
            namespace["sub_llm"] = self._safe_ask_llm  # Alias

        return namespace

    def _safe_ask_llm(self, prompt: str, context_subset: Optional[str] = None) -> str:
        """
        Safely call the sub-LLM with a prompt.
        
        Args:
            prompt: The query to ask the sub-LLM
            context_subset: Optional subset of context to include with the query
            
        Returns:
            The LLM's response as a string
        """
        if not self.ask_llm_fn:
            return "[ERROR: Sub-LLM function not available]"
        
        try:
            full_prompt = prompt
            if context_subset:
                full_prompt = f"Context:\n{context_subset}\n\nQuery: {prompt}"
            
            response = self.ask_llm_fn(full_prompt)
            return response
        except Exception as e:
            return f"[ERROR calling sub-LLM: {str(e)}]"

    # ============== Helper Functions ==============

    def _peek(self, start: int = 0, length: int = 2000) -> str:
        """Peek at a portion of the context."""
        return self.context[start:start + length]

    def _chunk(self, chunk_size: int = 10000, overlap: int = 200) -> List[str]:
        """Split context into overlapping chunks."""
        chunks = []
        i = 0
        while i < len(self.context):
            end = min(i + chunk_size, len(self.context))
            chunks.append(self.context[i:end])
            i += chunk_size - overlap
        return chunks

    def _grep(self, pattern: str, ignore_case: bool = True) -> List[str]:
        """Find lines matching a regex pattern."""
        flags = re.IGNORECASE if ignore_case else 0
        lines = self.context.split('\n')
        matched = [line for line in lines if re.search(pattern, line, flags)]
        return matched

    def _count_lines(self) -> int:
        """Count the number of lines in the context."""
        return len(self.context.split('\n'))

    def _head(self, n: int = 10) -> str:
        """Get the first n lines of the context."""
        lines = self.context.split('\n')
        return '\n'.join(lines[:n])

    def _tail(self, n: int = 10) -> str:
        """Get the last n lines of the context."""
        lines = self.context.split('\n')
        return '\n'.join(lines[-n:])

    # ============== Execution ==============

    def _safe_import(self, name: str) -> Any:
        """Safe import that only allows whitelisted modules."""
        if self.restricted and name not in self.allowed_imports:
            raise ImportError(f"Import of '{name}' is not allowed for security reasons")
        return __import__(name)

    def execute(self, code: str) -> REPLResult:
        """
        Execute Python code in the REPL environment.

        Args:
            code: Python code to execute

        Returns:
            REPLResult with success status, output, and any errors
        """
        import time
        start_time = time.time()
        
        # Security check for restricted mode
        if self.restricted:
            for name in self.RESTRICTED_NAMES:
                if re.search(rf'\b{name}\b', code):
                    return REPLResult(
                        success=False,
                        output="",
                        error=f"Security error: '{name}' is not allowed",
                        execution_time=0.0,
                    )
            
            # Check for dangerous imports
            import_pattern = r'import\s+(\w+)|from\s+(\w+)'
            for match in re.finditer(import_pattern, code):
                module_name = match.group(1) or match.group(2)
                if module_name in self.RESTRICTED_MODULES:
                    return REPLResult(
                        success=False,
                        output="",
                        error=f"Security error: Import of '{module_name}' is not allowed",
                        execution_time=0.0,
                    )

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Execute the code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to execute as expression first (for single-line evaluations)
                try:
                    # Check if it's a simple expression
                    result = eval(code, self.namespace)
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    # Execute as statements
                    exec(code, self.namespace)

            output = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # Combine outputs
            full_output = output
            if stderr:
                full_output += f"\n[stderr]: {stderr}"
            
            # Truncate if too long
            if len(full_output) > self.max_output_length:
                full_output = (
                    full_output[:self.max_output_length // 2] +
                    f"\n\n... [truncated {len(full_output) - self.max_output_length} characters] ...\n\n" +
                    full_output[-self.max_output_length // 2:]
                )

            execution_time = time.time() - start_time

            # Log to history
            self.history.append({
                "code": code,
                "output": full_output,
                "success": True,
                "time": execution_time,
            })

            return REPLResult(
                success=True,
                output=full_output,
                error=None,
                variables=dict(self.namespace),
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            
            # Log to history
            self.history.append({
                "code": code,
                "output": "",
                "error": str(e),
                "success": False,
                "time": execution_time,
            })

            return REPLResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"{type(e).__name__}: {str(e)}\n{error_trace}",
                variables=dict(self.namespace),
                execution_time=execution_time,
            )

    def update_context(self, new_context: str) -> None:
        """Update the context variable."""
        self.context = new_context
        self.namespace["context"] = new_context

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the namespace."""
        self.namespace[name] = value

    def get_variable(self, name: str) -> Any:
        """Get a variable from the namespace."""
        return self.namespace.get(name)

    def reset(self) -> None:
        """Reset the REPL to initial state."""
        self.history = []
        self.namespace = self._create_namespace()

    def get_available_functions(self) -> Dict[str, str]:
        """Get list of available helper functions with descriptions."""
        return {
            "peek(start, length)": "Peek at a portion of the context (default: first 2000 chars)",
            "chunk(chunk_size, overlap)": "Split context into overlapping chunks",
            "grep(pattern, ignore_case)": "Find lines matching a regex pattern",
            "count_lines()": "Count the number of lines in the context",
            "head(n)": "Get the first n lines of the context",
            "tail(n)": "Get the last n lines of the context",
            "ask_llm(prompt, context_subset)": "Call sub-LLM with a query (if available)",
        }


if __name__ == "__main__":
    # Quick test
    print("Testing REPL environment...")
    
    test_context = """
Line 1: Hello world
Line 2: User ID: 12345 || Instance: What is Python?
Line 3: User ID: 67890 || Instance: How old is the Earth?
Line 4: User ID: 12345 || Instance: Define recursion.
Line 5: Goodbye world
"""
    
    repl = REPL(context=test_context)
    
    # Test peek
    result = repl.execute("peek(0, 50)")
    print(f"Peek result: {result.output}")
    
    # Test grep
    result = repl.execute("grep('User ID')")
    print(f"Grep result: {result.output}")
    
    # Test count_lines
    result = repl.execute("count_lines()")
    print(f"Line count: {result.output}")
    
    # Test custom code
    result = repl.execute("""
matching_lines = grep('12345')
print(f"Found {len(matching_lines)} matching lines")
for line in matching_lines:
    print(line)
""")
    print(f"Custom code result: {result.output}")
