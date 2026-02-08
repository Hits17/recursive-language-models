#!/usr/bin/env python3
"""
RLM Main Example - Needle in a Haystack

This example demonstrates the RLM (Recursive Language Model) framework 
using your local Ollama service with Qwen3.

The example creates a "needle in a haystack" scenario where a random 
number is hidden within ~1M lines of random text, and the RLM must find it.
"""

import os
import random
import string
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from rlm import RLM


def generate_random_text(num_lines: int = 1000, min_words: int = 5, max_words: int = 15) -> str:
    """Generate random text lines."""
    lines = []
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
        "data", "analysis", "machine", "learning", "neural", "network",
        "artificial", "intelligence", "algorithm", "computer", "science",
        "python", "programming", "code", "function", "variable", "class",
        "object", "method", "return", "value", "string", "integer", "float",
    ]
    
    for _ in range(num_lines):
        num_words = random.randint(min_words, max_words)
        line = " ".join(random.choice(words) for _ in range(num_words))
        lines.append(line)
    
    return "\n".join(lines)


def needle_in_haystack_example():
    """
    Example: Find a hidden number in a sea of random text.
    
    This demonstrates how RLM can efficiently search through
    large contexts using its REPL capabilities.
    """
    print("=" * 60)
    print("RLM Example: Needle in a Haystack")
    print("=" * 60)
    
    # Generate the "haystack" - a large body of random text
    num_lines = 10000  # Approximately 100k characters
    print(f"\nGenerating haystack with {num_lines} lines...")
    haystack = generate_random_text(num_lines)
    
    # Generate the "needle" - a secret number hidden in a random line
    secret_number = random.randint(100000, 999999)
    needle_line = random.randint(num_lines // 4, 3 * num_lines // 4)
    needle = f"THE_SECRET_NUMBER_IS_{secret_number}"
    
    # Insert the needle
    lines = haystack.split("\n")
    lines[needle_line] = f"HIDDEN: {needle}"
    context = "\n".join(lines)
    
    print(f"Context size: {len(context):,} characters")
    print(f"Lines: {len(lines):,}")
    print(f"Secret number hidden at line {needle_line}: {secret_number}")
    print("-" * 60)
    
    # Initialize RLM
    rlm = RLM(
        verbose=True,
        max_iterations=10,
    )
    
    # Query the RLM
    query = """There is a hidden secret number in the context. 
    It appears in a line that starts with 'HIDDEN:' and contains 'THE_SECRET_NUMBER_IS_' followed by the number.
    Find this number and return it."""
    
    print("\n[Starting RLM Query]")
    print(f"Query: {query}")
    print("-" * 60)
    
    result = rlm.completion(query=query, context=context)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"RLM Response: {result.response}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.total_iterations}")
    print(f"Tokens used: {result.total_tokens}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Actual secret number: {secret_number}")
    
    # Check if the answer is correct
    if str(secret_number) in result.response:
        print("\n✅ SUCCESS! The RLM found the correct number!")
    else:
        print("\n❌ The RLM did not find the correct number.")
    
    return result


def multi_hop_example():
    """
    Example: Multi-hop reasoning over structured data.
    
    This demonstrates how RLM can handle queries that require
    aggregating information from multiple parts of the context.
    """
    print("\n" + "=" * 60)
    print("RLM Example: Multi-hop Query")
    print("=" * 60)
    
    # Create structured data with relationships
    context = """
# Product Database

## Products
- Product ID: P001 | Name: Widget A | Category: Electronics | Price: $50 | Manager: Alice
- Product ID: P002 | Name: Widget B | Category: Electronics | Price: $75 | Manager: Bob
- Product ID: P003 | Name: Gadget X | Category: Home | Price: $30 | Manager: Alice
- Product ID: P004 | Name: Gadget Y | Category: Home | Price: $45 | Manager: Carol
- Product ID: P005 | Name: Device Z | Category: Electronics | Price: $100 | Manager: Carol

## Sales Data (Q1 2024)
- Sale ID: S001 | Product: P001 | Quantity: 100 | Date: 2024-01-15
- Sale ID: S002 | Product: P002 | Quantity: 50 | Date: 2024-01-20
- Sale ID: S003 | Product: P003 | Quantity: 200 | Date: 2024-02-10
- Sale ID: S004 | Product: P001 | Quantity: 75 | Date: 2024-02-15
- Sale ID: S005 | Product: P005 | Quantity: 30 | Date: 2024-02-28
- Sale ID: S006 | Product: P004 | Quantity: 150 | Date: 2024-03-05
- Sale ID: S007 | Product: P002 | Quantity: 80 | Date: 2024-03-10
- Sale ID: S008 | Product: P003 | Quantity: 85 | Date: 2024-03-15

## Manager Information
- Name: Alice | Department: Sales | Bonus Rate: 5%
- Name: Bob | Department: Marketing | Bonus Rate: 3%
- Name: Carol | Department: Operations | Bonus Rate: 4%
"""

    query = """Calculate the total revenue generated by products managed by Alice.
    
    To do this, you need to:
    1. Find which products Alice manages
    2. Find all sales of those products
    3. Calculate revenue for each sale (quantity × price)
    4. Sum up the total revenue
    
    Return the final number."""

    print(f"Context size: {len(context)} characters")
    print("-" * 60)
    
    rlm = RLM(verbose=True, max_iterations=15)
    result = rlm.completion(query=query, context=context)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"RLM Response: {result.response}")
    print(f"Iterations: {result.total_iterations}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    # Calculate expected answer
    # Alice manages P001 ($50) and P003 ($30)
    # P001 sales: 100 + 75 = 175 units = $8,750
    # P003 sales: 200 + 85 = 285 units = $8,550
    # Total: $17,300
    print(f"\nExpected answer: $17,300")
    
    return result


def simple_counting_example():
    """
    Example: Simple counting task over data.
    
    Similar to the OOLONG benchmark style queries.
    """
    print("\n" + "=" * 60)
    print("RLM Example: Counting Task")  
    print("=" * 60)
    
    # Generate user data similar to OOLONG benchmark
    lines = []
    user_ids = [12345, 67890, 11111, 22222, 33333]
    question_types = ["entity", "description", "abbreviation", "human", "location", "numeric"]
    
    for i in range(500):
        user_id = random.choice(user_ids)
        q_type = random.choice(question_types)
        question = f"Sample question #{i} about {q_type} topic"
        lines.append(f"Date: 2024-{random.randint(1,12):02d}-{random.randint(1,28):02d} || User: {user_id} || Type: {q_type} || Question: {question}")
    
    context = "\n".join(lines)
    
    query = """Count how many data points are associated with User ID 12345 AND have Type 'entity'.
    Give your answer as a single number."""
    
    print(f"Context: {len(lines)} entries")
    print("-" * 60)
    
    rlm = RLM(verbose=True, max_iterations=10)
    result = rlm.completion(query=query, context=context)
    
    # Calculate expected answer
    expected = sum(1 for line in lines if "User: 12345" in line and "Type: entity" in line)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"RLM Response: {result.response}")
    print(f"Expected answer: {expected}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    return result


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# RECURSIVE LANGUAGE MODELS (RLM) - DEMONSTRATION")
    print("# Using local Ollama with Qwen3")
    print("#" * 60)
    
    # Check if Ollama is available
    try:
        from rlm import OllamaClient
        client = OllamaClient()
        print(f"\n✅ Connected to Ollama at {client.host}")
        print(f"✅ Using model: {client.model_name}")
    except Exception as e:
        print(f"\n❌ Error connecting to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return
    
    # Run examples
    print("\n" + "-" * 60)
    print("Select an example to run:")
    print("1. Needle in a Haystack (find hidden number)")
    print("2. Multi-hop Query (calculate revenue)")
    print("3. Counting Task (OOLONG-style)")
    print("4. Run all examples")
    print("-" * 60)
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        needle_in_haystack_example()
    elif choice == "2":
        multi_hop_example()
    elif choice == "3":
        simple_counting_example()
    elif choice == "4":
        needle_in_haystack_example()
        multi_hop_example()
        simple_counting_example()
    else:
        print("Invalid choice. Running needle in haystack example...")
        needle_in_haystack_example()


if __name__ == "__main__":
    main()
