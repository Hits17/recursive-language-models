"""
Enhanced RLM - Streaming Output & Progress

This module provides real-time streaming capabilities:
- Live token-by-token output
- Progress callbacks for long operations  
- Early termination when answer found

Improves UX significantly for long-running queries.
"""

import os
import sys
import time
import json
import requests
from typing import Optional, List, Dict, Any, Callable, Generator
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from dotenv import load_dotenv

load_dotenv()

console = Console()


@dataclass
class StreamingResult:
    """Result with streaming metadata."""
    content: str
    chunks_received: int
    total_time: float
    early_terminated: bool = False


class StreamingOllamaClient:
    """
    Streaming client for Ollama with real-time output.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def stream_completion(
        self,
        messages: List[Dict[str, str]] | str,
        on_token: Optional[Callable[[str], None]] = None,
        stop_condition: Optional[Callable[[str], bool]] = None,
        **kwargs
    ) -> Generator[str, None, StreamingResult]:
        """
        Stream completion tokens in real-time.
        
        Args:
            messages: The messages to send
            on_token: Callback for each token received
            stop_condition: Optional function to check if we should stop early
            **kwargs: Additional parameters
            
        Yields:
            Tokens as they arrive
            
        Returns:
            StreamingResult with full content and metadata
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 4096),
            },
        }

        start_time = time.time()
        full_content = ""
        chunks_received = 0
        early_terminated = False

        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=300,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        token = data["message"].get("content", "")
                        full_content += token
                        chunks_received += 1
                        
                        if on_token:
                            on_token(token)
                        
                        yield token
                        
                        # Check stop condition
                        if stop_condition and stop_condition(full_content):
                            early_terminated = True
                            break
                    
                    if data.get("done", False):
                        break

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        return StreamingResult(
            content=full_content,
            chunks_received=chunks_received,
            total_time=time.time() - start_time,
            early_terminated=early_terminated,
        )

    def completion_with_live_display(
        self,
        messages: List[Dict[str, str]] | str,
        title: str = "LLM Response",
    ) -> str:
        """
        Show completion with live updating rich display.
        """
        full_content = ""
        
        with Live(Panel("Waiting for response...", title=title), refresh_per_second=10) as live:
            for token in self.stream_completion(messages):
                full_content += token
                # Update display with current content
                live.update(Panel(
                    Markdown(full_content[-2000:]),  # Show last 2000 chars
                    title=title,
                    subtitle=f"{len(full_content)} chars"
                ))
        
        return full_content


class ProgressTracker:
    """
    Track and display progress for multi-step RLM operations.
    """

    def __init__(self, title: str = "RLM Processing"):
        self.title = title
        self.console = Console()
        self.steps: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def add_step(self, name: str, status: str = "pending"):
        """Add a step to track."""
        self.steps.append({
            "name": name,
            "status": status,
            "start_time": None,
            "end_time": None,
        })

    def start_step(self, name: str):
        """Mark a step as started."""
        for step in self.steps:
            if step["name"] == name:
                step["status"] = "running"
                step["start_time"] = time.time()
                self._display()
                break

    def complete_step(self, name: str, result: str = ""):
        """Mark a step as complete."""
        for step in self.steps:
            if step["name"] == name:
                step["status"] = "complete"
                step["end_time"] = time.time()
                step["result"] = result
                self._display()
                break

    def fail_step(self, name: str, error: str = ""):
        """Mark a step as failed."""
        for step in self.steps:
            if step["name"] == name:
                step["status"] = "failed"
                step["end_time"] = time.time()
                step["error"] = error
                self._display()
                break

    def _display(self):
        """Display current progress."""
        self.console.clear()
        self.console.print(f"\n[bold cyan]{self.title}[/bold cyan]")
        self.console.print("-" * 40)
        
        for step in self.steps:
            status_icon = {
                "pending": "⏳",
                "running": "🔄",
                "complete": "✅",
                "failed": "❌",
            }.get(step["status"], "?")
            
            elapsed = ""
            if step["start_time"]:
                end = step["end_time"] or time.time()
                elapsed = f" ({end - step['start_time']:.1f}s)"
            
            self.console.print(f"  {status_icon} {step['name']}{elapsed}")
            
            if step.get("result"):
                self.console.print(f"      └─ {step['result'][:50]}...")
        
        total_elapsed = time.time() - self.start_time
        self.console.print("-" * 40)
        self.console.print(f"Total time: {total_elapsed:.1f}s")


def stream_demo():
    """Demonstrate streaming output."""
    print("=" * 60)
    print("DEMO: Streaming Output")
    print("=" * 60)
    
    client = StreamingOllamaClient()
    
    print("\n[Streaming response token by token...]\n")
    
    # Simple streaming with live output
    full_response = ""
    for token in client.stream_completion(
        "Write a haiku about recursion.",
        on_token=lambda t: print(t, end="", flush=True),
    ):
        full_response += token
    
    print("\n")
    print(f"\n[Complete - {len(full_response)} characters received]")


def progress_demo():
    """Demonstrate progress tracking."""
    print("\n" + "=" * 60)
    print("DEMO: Progress Tracking")
    print("=" * 60)
    
    tracker = ProgressTracker("Multi-Step RLM Query")
    
    # Define steps
    tracker.add_step("Initialize REPL")
    tracker.add_step("Analyze context structure")
    tracker.add_step("Execute search query")
    tracker.add_step("Process results")
    tracker.add_step("Generate final answer")
    
    # Simulate execution
    import time
    
    tracker.start_step("Initialize REPL")
    time.sleep(0.5)
    tracker.complete_step("Initialize REPL", "REPL ready with 1024 chars loaded")
    
    tracker.start_step("Analyze context structure")
    time.sleep(0.3)
    tracker.complete_step("Analyze context structure", "Found 50 entries")
    
    tracker.start_step("Execute search query")
    time.sleep(0.8)
    tracker.complete_step("Execute search query", "4 matches found")
    
    tracker.start_step("Process results")
    time.sleep(0.4)
    tracker.complete_step("Process results", "Aggregated results")
    
    tracker.start_step("Generate final answer")
    time.sleep(0.6)
    tracker.complete_step("Generate final answer", "Answer: 4")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    stream_demo()
    progress_demo()
