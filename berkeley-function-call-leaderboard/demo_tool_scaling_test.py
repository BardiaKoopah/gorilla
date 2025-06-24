#!/usr/bin/env python3
"""
Demo Tool Scaling Test

This script demonstrates the tool scaling test concept by showing:
1. How test cases are generated with different numbers of tools
2. How the solution tool is placed at different positions
3. Sample analysis of results

This is a standalone demo that doesn't require the full BFCL installation.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
random.seed(42)

def load_simple_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the simple test data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_all_tools(simple_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract all unique tools from the simple dataset."""
    tools = []
    seen_names = set()
    
    for entry in simple_data:
        for func in entry['function']:
            if func['name'] not in seen_names:
                tools.append(func)
                seen_names.add(func['name'])
    
    return tools

def create_tool_set(solution_tool: Dict[str, Any], all_tools: List[Dict[str, Any]], 
                   total_tools: int, solution_position: int) -> List[Dict[str, Any]]:
    """Create a tool set with the solution tool at the specified position."""
    if total_tools == 1:
        return [solution_tool]
    
    # Adjust position if it's beyond the total number of tools
    actual_position = min(solution_position, total_tools)
    
    # Get distractor tools (excluding the solution tool)
    distractor_tools = [tool for tool in all_tools if tool['name'] != solution_tool['name']]
    
    # If we need more tools than available, repeat some
    if total_tools - 1 > len(distractor_tools):
        multiplier = ((total_tools - 1) // len(distractor_tools)) + 1
        distractor_tools = distractor_tools * multiplier
    
    # Take the first 'total_tools - 1' distractor tools for consistency
    selected_distractors = distractor_tools[:total_tools - 1]
    
    # Create the tool list
    tools = []
    
    # Add tools before the solution position
    tools.extend(selected_distractors[:actual_position - 1])
    
    # Add the solution tool
    tools.append(solution_tool)
    
    # Add remaining distractor tools
    remaining_distractors = selected_distractors[actual_position - 1:]
    tools.extend(remaining_distractors)
    
    return tools[:total_tools]

def demonstrate_test_generation():
    """Demonstrate how test cases are generated."""
    print("=== Tool Scaling Test Generation Demo ===\n")
    
    # Load sample data
    simple_data_path = "bfcl_eval/data/BFCL_v3_simple.json"
    if not Path(simple_data_path).exists():
        print(f"Error: {simple_data_path} not found. Please run this from the BFCL directory.")
        return
    
    simple_data = load_simple_data(simple_data_path)
    all_tools = extract_all_tools(simple_data)
    
    print(f"Loaded {len(simple_data)} test cases")
    print(f"Found {len(all_tools)} unique tools")
    
    # Demo configurations
    tool_counts = [1, 2, 5, 10, 20]
    solution_positions = [1, 5, 20]
    
    # Take first test case as example
    example_case = simple_data[0]
    solution_tool = example_case['function'][0]
    
    print(f"\nExample test case: {example_case['id']}")
    print(f"Question: {example_case['question'][0][0]['content']}")
    print(f"Solution tool: {solution_tool['name']}")
    
    print("\n=== Generated Test Variations ===")
    
    for pos in solution_positions:
        print(f"\n--- Solution tool at position {pos} ---")
        for count in tool_counts:
            if pos > count:
                print(f"  {count} tools: Skipped (position > tool count)")
                continue
            
            # Generate tool set
            tool_set = create_tool_set(solution_tool, all_tools, count, pos)
            
            # Find actual position of solution tool
            actual_pos = next(i+1 for i, tool in enumerate(tool_set) if tool['name'] == solution_tool['name'])
            
            print(f"  {count} tools: Solution at position {actual_pos}")
            print(f"    Tools: {[tool['name'] for tool in tool_set]}")

def simulate_results():
    """Simulate evaluation results to demonstrate analysis."""
    print("\n=== Simulated Results Analysis ===\n")
    
    # Simulate realistic results where:
    # - Performance generally decreases with more tools
    # - Position 1 performs better than later positions
    # - Some randomness to make it realistic
    
    tool_counts = [1, 2, 5, 10, 20, 40, 80]
    solution_positions = [1, 5, 20, 50]
    
    results = []
    
    for pos in solution_positions:
        for count in tool_counts:
            if pos > count:
                continue
            
            # Simulate accuracy based on realistic patterns
            base_accuracy = 0.95  # Start high
            
            # Decrease with more tools (tool confusion effect)
            tool_penalty = (count - 1) * 0.01
            
            # Decrease with later positions (position bias effect)
            position_penalty = (pos - 1) * 0.005
            
            # Add some randomness
            noise = random.uniform(-0.05, 0.05)
            
            accuracy = max(0.0, min(1.0, base_accuracy - tool_penalty - position_penalty + noise))
            
            results.append({
                'tool_count': count,
                'solution_position': pos,
                'accuracy': accuracy,
                'category': f'simple_tools_{count}_pos_{pos}'
            })
    
    df = pd.DataFrame(results)
    
    # Display results table
    print("Simulated Results:")
    print(df.pivot(index='solution_position', columns='tool_count', values='accuracy').round(3))
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    for pos in sorted(df['solution_position'].unique()):
        pos_data = df[df['solution_position'] == pos]
        plt.plot(pos_data['tool_count'], pos_data['accuracy'], 
                marker='o', linewidth=2, markersize=8, 
                label=f'Position {pos}')
    
    plt.xlabel('Number of Tools', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Tool Scaling Performance (Simulated)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    # Save plot
    output_file = 'tool_scaling_demo_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    
    # Analysis insights
    print("\n=== Key Insights from Simulated Results ===")
    
    # Tool count effect
    pos1_data = df[df['solution_position'] == 1]
    accuracy_1_tool = pos1_data[pos1_data['tool_count'] == 1]['accuracy'].iloc[0]
    accuracy_80_tools = pos1_data[pos1_data['tool_count'] == 80]['accuracy'].iloc[0]
    
    print(f"1. Tool Count Effect:")
    print(f"   - 1 tool: {accuracy_1_tool:.3f} accuracy")
    print(f"   - 80 tools: {accuracy_80_tools:.3f} accuracy")
    print(f"   - Performance drop: {accuracy_1_tool - accuracy_80_tools:.3f}")
    
    # Position effect
    count10_data = df[df['tool_count'] == 10]
    pos_effects = count10_data.groupby('solution_position')['accuracy'].mean()
    
    print(f"\n2. Position Effect (10 tools):")
    for pos, acc in pos_effects.items():
        print(f"   - Position {pos}: {acc:.3f} accuracy")
    
    # Best/worst configurations
    best_config = df.loc[df['accuracy'].idxmax()]
    worst_config = df.loc[df['accuracy'].idxmin()]
    
    print(f"\n3. Best Configuration: {best_config['tool_count']} tools, position {best_config['solution_position']} ({best_config['accuracy']:.3f})")
    print(f"4. Worst Configuration: {worst_config['tool_count']} tools, position {worst_config['solution_position']} ({worst_config['accuracy']:.3f})")
    
    return df

def show_usage_instructions():
    """Show instructions for using the actual tool scaling test."""
    print("\n=== How to Use the Tool Scaling Test ===\n")
    
    print("1. Generate test files:")
    print("   python tool_scaling_test.py --action generate --max-test-cases 50")
    
    print("\n2. Copy test files to BFCL data directory:")
    print("   cp tool_scaling_tests/BFCL_v3_*.json bfcl_eval/data/")
    
    print("\n3. Update category mapping (already done in this demo)")
    
    print("\n4. Run evaluations for a specific model:")
    print("   # Example with a single configuration")
    print("   bfcl generate --model gpt-4o --test-category simple_tools_5_pos_1")
    print("   bfcl evaluate --model gpt-4o --test-category simple_tools_5_pos_1")
    
    print("\n5. Run complete evaluation suite:")
    print("   python run_tool_scaling_evaluation.py --model gpt-4o --quick-test")
    
    print("\n6. Analyze results:")
    print("   python run_tool_scaling_evaluation.py --model gpt-4o --analysis-only")
    
    print("\nTest Configurations Generated:")
    tool_counts = [1, 2, 5, 10, 20, 40, 80]
    solution_positions = [1, 5, 20, 50]
    
    count = 0
    for pos in solution_positions:
        for tools in tool_counts:
            if pos <= tools:
                count += 1
                print(f"  - simple_tools_{tools}_pos_{pos}")
    
    print(f"\nTotal configurations: {count}")

def main():
    """Run the complete demo."""
    print("Tool Scaling Test for BFCL - Demonstration")
    print("=" * 50)
    
    # Demo 1: Show test generation
    demonstrate_test_generation()
    
    # Demo 2: Simulate and analyze results
    try:
        results_df = simulate_results()
    except ImportError as e:
        print(f"Skipping visualization demo due to missing dependency: {e}")
        print("Install matplotlib and pandas to see the full demo.")
    
    # Demo 3: Show usage instructions
    show_usage_instructions()
    
    print("\n" + "=" * 50)
    print("Demo complete! The tool scaling test framework is ready to use.")

if __name__ == "__main__":
    main()