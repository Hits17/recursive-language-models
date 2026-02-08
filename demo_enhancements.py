#!/usr/bin/env python3
"""
RLM Enhancements Demo

This script demonstrates all the enhanced features:
1. RAG Integration - Pre-filtering with semantic search
2. Budget Controls - Token and cost management
3. Streaming Output - Real-time responses
4. Progress Tracking - Visual progress for multi-step operations

Run with: python demo_enhancements.py
"""

import time
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()


def demo_header(title: str, description: str):
    """Print a demo section header."""
    console.print()
    console.print(Panel(
        f"[bold]{description}[/bold]",
        title=f"🚀 {title}",
        border_style="cyan"
    ))
    console.print()


# ============== Demo 1: RAG Integration ==============

def demo_rag_integration():
    """Demonstrate RAG-enhanced context filtering."""
    demo_header(
        "RAG Integration",
        "Pre-filter large contexts using semantic search before applying RLM reasoning"
    )
    
    from rlm.rag import RAGEnhancedRLM
    
    # Create a multi-topic context (simulating a large document)
    context = """
    SECTION 1: COMPANY HISTORY
    Acme Corporation was founded in 1985 by John Smith.
    The company started as a small hardware manufacturer.
    By 1990, Acme had expanded to 500 employees.
    The headquarters is located in San Francisco, California.
    
    SECTION 2: PRODUCTS AND SERVICES
    Our flagship product is the Widget Pro 3000.
    The Widget Pro 3000 costs $499 and has 10GB of RAM.
    We also offer cloud services starting at $29/month.
    Enterprise customers get 24/7 support.
    
    SECTION 3: FINANCIAL INFORMATION
    Q1 2024 Revenue: $50 million
    Q2 2024 Revenue: $62 million  
    Q3 2024 Revenue: $58 million
    Q4 2024 Revenue: $75 million
    Total 2024 Revenue: $245 million
    
    SECTION 4: EMPLOYEE INFORMATION
    Current headcount: 2,500 employees
    Engineering team: 800 people
    Sales team: 600 people
    Marketing team: 300 people
    The CEO is Sarah Johnson, appointed in 2020.
    
    SECTION 5: SUSTAINABILITY INITIATIVES
    We committed to carbon neutrality by 2030.
    Solar panels installed at HQ in 2022.
    Electric vehicle fleet for all deliveries.
    Recycling program saved 100 tons of waste.
    """
    
    console.print(f"[dim]Original context: {len(context)} characters[/dim]")
    
    # Initialize RAG
    rag = RAGEnhancedRLM(chunk_size=500, top_k=2)
    num_chunks = rag.index_context(context)
    console.print(f"[dim]Indexed into {num_chunks} chunks[/dim]\n")
    
    # Test queries
    queries = [
        "What was the total revenue in 2024?",
        "Who is the CEO?",
        "How much does the Widget Pro cost?",
    ]
    
    table = Table(title="RAG Search Results")
    table.add_column("Query", style="cyan")
    table.add_column("Top Match Score", style="green")
    table.add_column("Context Compression", style="yellow")
    
    for query in queries:
        results = rag.retrieve(query)
        filtered, _ = rag.get_filtered_context(query)
        
        top_score = results[0].score if results else 0
        compression = f"{len(filtered)/len(context)*100:.0f}%"
        
        table.add_row(query, f"{top_score:.3f}", compression)
    
    console.print(table)
    
    console.print("\n[green]✅ RAG reduces context by ~60-80% while keeping relevant info![/green]")


# ============== Demo 2: Budget Controls ==============

def demo_budget_controls():
    """Demonstrate budget management."""
    demo_header(
        "Budget Controls",
        "Manage token usage and costs with automatic limits and warnings"
    )
    
    from rlm.budget import BudgetManager, BudgetConfig
    
    # Create a constrained budget
    config = BudgetConfig(
        max_total_tokens=10000,
        max_iterations=5,
        max_query_time=60.0,
    )
    
    budget = BudgetManager(config)
    
    console.print("[bold]Simulating RLM operations with budget tracking...[/bold]\n")
    
    table = Table(title="Budget Usage Simulation")
    table.add_column("Iteration", style="cyan")
    table.add_column("Tokens Used", style="green")
    table.add_column("Remaining", style="yellow")
    table.add_column("Status", style="magenta")
    
    for i in range(7):  # Try more iterations than allowed
        # Check budget
        reason = budget.check_can_proceed()
        if reason:
            table.add_row(
                f"{i+1}", 
                f"{budget.usage.tokens_used:,}", 
                "0",
                f"🛑 {reason.value}"
            )
            break
        
        # Simulate token usage
        budget.record_usage(
            input_tokens=1500,
            output_tokens=500,
            is_iteration=True,
        )
        
        remaining = budget.get_remaining()
        table.add_row(
            f"{i+1}",
            f"{budget.usage.tokens_used:,}",
            f"{remaining['tokens']:,}",
            "✅ OK"
        )
    
    console.print(table)
    console.print(f"\n[dim]{budget.format_summary()}[/dim]")


# ============== Demo 3: Streaming Output ==============

def demo_streaming():
    """Demonstrate streaming output."""
    demo_header(
        "Streaming Output", 
        "Real-time token-by-token output for better UX"
    )
    
    from rlm.streaming import StreamingOllamaClient
    
    client = StreamingOllamaClient()
    
    console.print("[bold]Streaming a response from Qwen3...[/bold]\n")
    console.print("[dim]Each character appears as it's generated:[/dim]\n")
    
    console.print("─" * 50)
    
    full_response = ""
    token_count = 0
    start = time.time()
    
    for token in client.stream_completion(
        "Explain recursion in one sentence.",
        on_token=lambda t: console.print(t, end="", highlight=False),
    ):
        full_response += token
        token_count += 1
    
    elapsed = time.time() - start
    
    console.print("\n" + "─" * 50)
    console.print(f"\n[dim]Received {token_count} tokens in {elapsed:.1f}s[/dim]")
    console.print("[green]✅ Streaming provides instant feedback![/green]")


# ============== Demo 4: Progress Tracking ==============

def demo_progress_tracking():
    """Demonstrate progress tracking."""
    demo_header(
        "Progress Tracking",
        "Visual progress for multi-step RLM operations"
    )
    
    from rlm.streaming import ProgressTracker
    
    tracker = ProgressTracker("Multi-Step RLM Demo")
    
    steps = [
        ("Initialize REPL", 0.3, "REPL ready"),
        ("Index context", 0.5, "12 chunks created"),
        ("Retrieve relevant chunks", 0.4, "5 matches found"),
        ("Execute REPL code", 0.6, "Code executed successfully"),
        ("Generate answer", 0.8, "Answer: 42"),
    ]
    
    # Add all steps
    for name, _, _ in steps:
        tracker.add_step(name)
    
    # Execute steps
    for name, duration, result in steps:
        tracker.start_step(name)
        time.sleep(duration)
        tracker.complete_step(name, result)
    
    console.print("\n[green]✅ Progress tracking provides visibility into RLM operations![/green]")


# ============== Main Demo Runner ==============

def main():
    """Run all enhancement demos."""
    console.print(Panel(
        "[bold cyan]RLM Enhancement Demonstrations[/bold cyan]\n\n"
        "This demo showcases the production-ready enhancements:\n"
        "• RAG Integration - Smart context filtering\n"
        "• Budget Controls - Token & cost management\n"
        "• Streaming - Real-time output\n"
        "• Progress Tracking - Visual feedback",
        title="🎯 Enhanced RLM",
        border_style="blue"
    ))
    
    demos = [
        ("1", "RAG Integration", demo_rag_integration),
        ("2", "Budget Controls", demo_budget_controls),
        ("3", "Streaming Output", demo_streaming),
        ("4", "Progress Tracking", demo_progress_tracking),
    ]
    
    console.print("\n[bold]Select a demo to run:[/bold]")
    for num, name, _ in demos:
        console.print(f"  {num}. {name}")
    console.print("  5. Run ALL demos")
    console.print("  q. Quit")
    
    choice = input("\nEnter choice: ").strip().lower()
    
    if choice == "q":
        return
    elif choice == "5":
        for _, name, func in demos:
            try:
                func()
            except Exception as e:
                console.print(f"[red]Error in {name}: {e}[/red]")
            console.print()
    elif choice in ["1", "2", "3", "4"]:
        idx = int(choice) - 1
        try:
            demos[idx][2]()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    else:
        console.print("[yellow]Invalid choice, running RAG demo...[/yellow]")
        demo_rag_integration()
    
    console.print("\n[bold green]Demo complete![/bold green]")


if __name__ == "__main__":
    main()
