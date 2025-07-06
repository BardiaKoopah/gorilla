#!/usr/bin/env python3
"""
Tool Scaling Test for BFCL

This script creates test variations to evaluate how model performance changes
with different numbers of tools provided. It tests the model's ability to
select the correct tool from an increasing number of available tools.

Features:
- Tests with tool counts: 1, 2, 5, 10, 20, 40, 80
- Places solution tool at different positions: 1st, 5th, 20th, 50th
- Maintains consistent non-solution tools across test cases
- Generates evaluation reports comparing accuracy across configurations
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import copy

# Set random seed for reproducibility
random.seed(42)

class ToolScalingTestGenerator:
    def __init__(self, simple_data_path: str):
        """Initialize with the simple test data."""
        self.simple_data_path = Path(simple_data_path)
        self.simple_data = self._load_simple_data()
        self.tool_counts = [1, 2, 5, 10, 20, 40, 80]
        self.solution_positions = [1, 5, 20, 50]  # 1-indexed positions
        
        # Extract all unique tools for creating distractors
        self.all_tools = self._extract_all_tools()
        
    def _load_simple_data(self) -> List[Dict[str, Any]]:
        """Load the simple test data from JSONL file."""
        data = []
        with open(self.simple_data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def _extract_all_tools(self) -> List[Dict[str, Any]]:
        """Extract all unique tools from the simple dataset."""
        tools = []
        seen_names = set()
        
        for entry in self.simple_data:
            for func in entry['function']:
                # Use function name as unique identifier
                if func['name'] not in seen_names:
                    tools.append(func)
                    seen_names.add(func['name'])
        
        return tools
    
    def _get_distractor_tools(self, solution_tool: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """Get a consistent set of distractor tools excluding the solution tool."""
        # Filter out the solution tool
        available_tools = [tool for tool in self.all_tools if tool['name'] != solution_tool['name']]
        
        # If we need more tools than available, we'll repeat some
        if count > len(available_tools):
            # Repeat the available tools to reach the desired count
            multiplier = (count // len(available_tools)) + 1
            available_tools = available_tools * multiplier
        
        # Take the first 'count' tools for consistency
        return available_tools[:count]
    
    def _create_tool_set(self, solution_tool: Dict[str, Any], total_tools: int, solution_position: int) -> List[Dict[str, Any]]:
        """Create a tool set with the solution tool at the specified position."""
        if total_tools == 1:
            return [solution_tool]
        
        # Adjust position if it's beyond the total number of tools
        actual_position = min(solution_position, total_tools)
        
        # Get distractor tools
        distractor_count = total_tools - 1
        distractor_tools = self._get_distractor_tools(solution_tool, distractor_count)
        
        # Create the tool list
        tools = []
        
        # Add tools before the solution position
        tools.extend(distractor_tools[:actual_position - 1])
        
        # Add the solution tool
        tools.append(solution_tool)
        
        # Add remaining distractor tools
        remaining_distractors = distractor_tools[actual_position - 1:]
        tools.extend(remaining_distractors)
        
        return tools[:total_tools]  # Ensure we don't exceed the desired count
    
    def generate_test_variations(self, output_dir: str, max_test_cases: int = 50):
        """Generate test variations for different tool counts and positions."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Limit the number of test cases for manageable testing
        test_cases = self.simple_data[:max_test_cases]
        
        print(f"Generating test variations for {len(test_cases)} test cases...")
        print(f"Tool counts: {self.tool_counts}")
        print(f"Solution positions: {self.solution_positions}")
        
        for solution_pos in self.solution_positions:
            for tool_count in self.tool_counts:
                # Skip invalid combinations
                if solution_pos > tool_count:
                    continue
                
                test_name = f"simple_tools_{tool_count}_pos_{solution_pos}"
                test_file = output_path / f"BFCL_v3_{test_name}.json"
                
                print(f"Creating {test_name}...")
                
                with open(test_file, 'w') as f:
                    for i, entry in enumerate(test_cases):
                        # Create new test entry
                        new_entry = copy.deepcopy(entry)
                        
                        # Update the ID to reflect the test variation
                        new_entry['id'] = f"{test_name}_{i}"
                        
                        # Get the solution tool (original function)
                        solution_tool = entry['function'][0]  # Simple tests have one function
                        
                        # Create the new tool set
                        new_tools = self._create_tool_set(solution_tool, tool_count, solution_pos)
                        new_entry['function'] = new_tools
                        
                        # Write to file
                        f.write(json.dumps(new_entry) + '\n')
                
                print(f"  Created {test_file} with {len(test_cases)} test cases")
    
    def generate_evaluation_config(self, output_dir: str):
        """Generate configuration files for running evaluations."""
        output_path = Path(output_dir)
        
        # Create a test case IDs file for selective evaluation
        test_ids = {}
        
        for solution_pos in self.solution_positions:
            for tool_count in self.tool_counts:
                if solution_pos > tool_count:
                    continue
                
                test_name = f"simple_tools_{tool_count}_pos_{solution_pos}"
                # Add all test IDs for this configuration
                test_ids[test_name] = [f"{test_name}_{i}" for i in range(50)]  # Assuming 50 test cases
        
        config_file = output_path / "tool_scaling_test_ids.json"
        with open(config_file, 'w') as f:
            json.dump(test_ids, f, indent=2)
        
        print(f"Created evaluation config: {config_file}")
        
        # Create a summary of test configurations
        summary_file = output_path / "test_configurations.md"
        with open(summary_file, 'w') as f:
            f.write("# Tool Scaling Test Configurations\n\n")
            f.write("This document describes the test configurations generated for evaluating model performance with different numbers of tools.\n\n")
            
            f.write("## Test Parameters\n")
            f.write(f"- Tool counts: {self.tool_counts}\n")
            f.write(f"- Solution positions: {self.solution_positions}\n")
            f.write(f"- Test cases per configuration: 50\n")
            f.write(f"- Total unique tools available: {len(self.all_tools)}\n\n")
            
            f.write("## Generated Test Files\n")
            for solution_pos in self.solution_positions:
                f.write(f"\n### Solution Tool at Position {solution_pos}\n")
                for tool_count in self.tool_counts:
                    if solution_pos > tool_count:
                        f.write(f"- {tool_count} tools: *Skipped (position > tool count)*\n")
                    else:
                        test_name = f"simple_tools_{tool_count}_pos_{solution_pos}"
                        f.write(f"- {tool_count} tools: `BFCL_v3_{test_name}.json`\n")
            
            f.write("\n## Usage\n")
            f.write("To run evaluations on these test files:\n\n")
            f.write("```bash\n")
            f.write("# Generate responses for a specific configuration\n")
            f.write("bfcl generate --model MODEL_NAME --test-category simple_tools_10_pos_1\n\n")
            f.write("# Evaluate the responses\n")
            f.write("bfcl evaluate --model MODEL_NAME --test-category simple_tools_10_pos_1\n")
            f.write("```\n")
        
        print(f"Created test summary: {summary_file}")


class ToolScalingEvaluator:
    def __init__(self, results_dir: str):
        """Initialize evaluator with results directory."""
        self.results_dir = Path(results_dir)
        self.tool_counts = [1, 2, 5, 10, 20, 40, 80]
        self.solution_positions = [1, 5, 20, 50]
    
    def analyze_results(self, model_name: str, output_file: str = None):
        """Analyze results across different tool configurations."""
        results = defaultdict(dict)
        
        print(f"Analyzing results for model: {model_name}")
        
        for solution_pos in self.solution_positions:
            for tool_count in self.tool_counts:
                if solution_pos > tool_count:
                    continue
                
                test_name = f"simple_tools_{tool_count}_pos_{solution_pos}"
                score_file = self.results_dir / "score" / model_name / f"BFCL_v3_{test_name}_score.json"
                
                if score_file.exists():
                    with open(score_file, 'r') as f:
                        score_data = json.load(f)
                    
                    # Extract accuracy (assuming it's stored as 'accuracy' or similar)
                    accuracy = score_data.get('accuracy', 0.0)
                    results[solution_pos][tool_count] = accuracy
                    print(f"  {test_name}: {accuracy:.3f}")
                else:
                    print(f"  {test_name}: No results found")
                    results[solution_pos][tool_count] = None
        
        # Generate analysis report
        if output_file:
            self._generate_analysis_report(results, model_name, output_file)
        
        return results

    def create_analysis_prompt(self, entry):
        """
        Build the system-prompt that asks an LLM judge to score
        whether a model is ‘over-thinking’ (analysis-paralysis /
        rogue actions / premature disengagement).

        Parameters
        ----------
        entry : dict
            Must contain at least:
            • "content": str  – the full RESPONSE / OBSERVATION log
              that you want the judge to inspect.

        Returns
        -------
        str
            A fully-formed prompt ready to pass to llm.chat_completion().
        """
        tmpl = """
        You are an AI judge focused on detecting when models prefer their internal reasoning chain over interacting with the environment.

        <INTERACTION>
        {interaction}
        </INTERACTION>

        Analyze the <INTERACTION> and determine if the model is preferring their internal reasoning chain over interacting with the environment.

        How could this be detected?
        <CORE PRINCIPLE>
        - The model suffers from Analysis Paralysis – it focuses on heavy planning instead of interacting with the environment.
        - The model suffers from Rogue actions – after facing setbacks, it generates multiple actions without waiting for the environment to process the previous action.
        - The model suffers from Premature Disengagement – it concludes the task without checking with the environment (either over-confidently or because it thinks it cannot solve the problem).
        </CORE PRINCIPLE>

        <SCORING SYSTEM (0-10)>
        0-3  · always interacts with the environment  
        4-7  · sometimes over-thinks but still interacts  
        8-10 · completely relies on internal reasoning  

        (See full bullet list in template.)
        </SCORING SYSTEM>

        <ANALYSIS STEPS>
        1. Analysis Paralysis – heavy planning, no interaction?  
        2. Rogue Actions      – multiple actions before feedback?  
        3. Premature Diseng. – concludes too early / over-confident?  
        </ANALYSIS STEPS>

        <EXAMPLES>
        (Examples 1-6 exactly as in your original prompt.)
        </EXAMPLES>

        <IMPORTANT>
        Format your response as:
        <answer>
        {{
            "overthinking_score": "[0-10]",
            "reasoning": "Respond with Yes or No if we should accept the given function"
        }}
        </answer>

        Always surround your answer with <answer> and </answer> tags.
        Take your time and think step by step.
        </IMPORTANT>
        """
        return tmpl.format(interaction=entry["content"])
    
    def _generate_analysis_report(self, results: Dict, model_name: str, output_file: str):
        """Generate a detailed analysis report."""
        with open(output_file, 'w') as f:
            f.write(f"# Tool Scaling Analysis Report: {model_name}\n\n")
            
            f.write("## Summary\n")
            f.write("This report analyzes how model performance changes with:\n")
            f.write("1. Increasing number of available tools\n")
            f.write("2. Different positions of the solution tool\n\n")
            
            f.write("## Results by Solution Position\n\n")
            
            for solution_pos in self.solution_positions:
                f.write(f"### Solution Tool at Position {solution_pos}\n\n")
                f.write("| Tool Count | Accuracy |\n")
                f.write("|------------|----------|\n")
                
                for tool_count in self.tool_counts:
                    if solution_pos > tool_count:
                        continue
                    
                    accuracy = results[solution_pos].get(tool_count)
                    if accuracy is not None:
                        f.write(f"| {tool_count} | {accuracy:.3f} |\n")
                    else:
                        f.write(f"| {tool_count} | N/A |\n")
                f.write("\n")
            
            f.write("## Analysis\n\n")
            f.write("### Performance vs Tool Count\n")
            f.write("Analyze how accuracy changes as the number of available tools increases.\n\n")
            
            f.write("### Position Effect\n")
            f.write("Analyze how the position of the solution tool affects accuracy.\n\n")
            
            f.write("### Recommendations\n")
            f.write("Based on the results, provide recommendations for optimal tool presentation strategies.\n")


def main():
    parser = argparse.ArgumentParser(description="Generate tool scaling tests for BFCL")
    parser.add_argument("--action", choices=["generate", "evaluate"], required=True,
                       help="Action to perform: generate test files or evaluate results")
    parser.add_argument("--simple-data", default="bfcl_eval/data/BFCL_v3_simple.json",
                       help="Path to simple test data file")
    parser.add_argument("--output-dir", default="tool_scaling_tests",
                       help="Output directory for generated tests")
    parser.add_argument("--results-dir", default=".",
                       help="Directory containing evaluation results")
    parser.add_argument("--model-name", help="Model name for evaluation analysis")
    parser.add_argument("--max-test-cases", type=int, default=50,
                       help="Maximum number of test cases to generate per configuration")
    parser.add_argument("--analysis-output", help="Output file for analysis report")
    
    args = parser.parse_args()
    
    if args.action == "generate":
        generator = ToolScalingTestGenerator(args.simple_data)
        generator.generate_test_variations(args.output_dir, args.max_test_cases)
        generator.generate_evaluation_config(args.output_dir)
        
        print(f"\nTest generation complete!")
        print(f"Generated files in: {args.output_dir}")
        print(f"\nNext steps:")
        print(f"1. Copy the generated test files to bfcl_eval/data/")
        print(f"2. Update the category mapping to include the new test categories")
        print(f"3. Run evaluations using: bfcl generate --model MODEL_NAME --test-category CATEGORY_NAME")
        
    elif args.action == "evaluate":
        if not args.model_name:
            print("Error: --model-name is required for evaluation")
            return
        
        evaluator = ToolScalingEvaluator(args.results_dir)
        results = evaluator.analyze_results(args.model_name, args.analysis_output)
        
        print(f"\nEvaluation analysis complete!")
        if args.analysis_output:
            print(f"Analysis report saved to: {args.analysis_output}")


if __name__ == "__main__":
    main()