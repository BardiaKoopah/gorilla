#!/usr/bin/env python3
"""
Tool Scaling Evaluation Runner

This script runs the complete tool scaling evaluation pipeline:
1. Generates model responses for all tool scaling test configurations
2. Evaluates the responses
3. Generates comprehensive analysis reports

Usage:
    python run_tool_scaling_evaluation.py --model MODEL_NAME [--quick-test]
"""

import argparse
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import time

class ToolScalingEvaluationRunner:
    def __init__(self, model_name: str, quick_test: bool = False):
        self.model_name = model_name
        self.quick_test = quick_test
        
        # Test configurations
        self.tool_counts = [1, 2, 5, 10, 20, 40, 80]
        self.solution_positions = [1, 5, 20, 50]
        
        # Get valid test categories (skip invalid position/count combinations)
        self.test_categories = []
        for pos in self.solution_positions:
            for count in self.tool_counts:
                if pos <= count:
                    self.test_categories.append(f"simple_tools_{count}_pos_{pos}")
        
        if quick_test:
            # For quick testing, only run a subset
            self.test_categories = [
                "simple_tools_1_pos_1",
                "simple_tools_5_pos_1", 
                "simple_tools_5_pos_5",
                "simple_tools_10_pos_1",
                "simple_tools_10_pos_5"
            ]
            print(f"Quick test mode: Running {len(self.test_categories)} configurations")
        else:
            print(f"Full test mode: Running {len(self.test_categories)} configurations")
    
    def run_generation(self):
        """Generate model responses for all test configurations."""
        print(f"\n=== Generating responses for {self.model_name} ===")
        
        for i, category in enumerate(self.test_categories, 1):
            print(f"\n[{i}/{len(self.test_categories)}] Generating responses for {category}...")
            
            cmd = [
                "bfcl", "generate",
                "--model", self.model_name,
                "--test-category", category,
                "--num-threads", "1"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    print(f"  ✓ Successfully generated responses for {category}")
                else:
                    print(f"  ✗ Failed to generate responses for {category}")
                    print(f"    Error: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"  ✗ Timeout generating responses for {category}")
            except Exception as e:
                print(f"  ✗ Error generating responses for {category}: {e}")
    
    def run_evaluation(self):
        """Evaluate generated responses for all test configurations."""
        print(f"\n=== Evaluating responses for {self.model_name} ===")
        
        for i, category in enumerate(self.test_categories, 1):
            print(f"\n[{i}/{len(self.test_categories)}] Evaluating {category}...")
            
            cmd = [
                "bfcl", "evaluate",
                "--model", self.model_name,
                "--test-category", category
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print(f"  ✓ Successfully evaluated {category}")
                else:
                    print(f"  ✗ Failed to evaluate {category}")
                    print(f"    Error: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"  ✗ Timeout evaluating {category}")
            except Exception as e:
                print(f"  ✗ Error evaluating {category}: {e}")
    
    def collect_results(self) -> Dict[str, Dict[str, float]]:
        """Collect evaluation results from score files."""
        print(f"\n=== Collecting results for {self.model_name} ===")
        
        results = {}
        score_dir = Path("score") / self.model_name
        
        for category in self.test_categories:
            score_file = score_dir / f"BFCL_v3_{category}_score.json"
            
            if score_file.exists():
                try:
                    with open(score_file, 'r') as f:
                        score_data = json.load(f)
                    
                    # Extract accuracy - the exact key might vary
                    accuracy = None
                    for key in ['accuracy', 'overall_accuracy', 'score']:
                        if key in score_data:
                            accuracy = score_data[key]
                            break
                    
                    if accuracy is None:
                        # Try to find any numeric value that looks like accuracy
                        for key, value in score_data.items():
                            if isinstance(value, (int, float)) and 0 <= value <= 1:
                                accuracy = value
                                break
                    
                    if accuracy is not None:
                        results[category] = {"accuracy": accuracy}
                        print(f"  ✓ {category}: {accuracy:.3f}")
                    else:
                        print(f"  ? {category}: Could not find accuracy in score file")
                        results[category] = {"accuracy": 0.0}
                        
                except Exception as e:
                    print(f"  ✗ Error reading {category}: {e}")
                    results[category] = {"accuracy": 0.0}
            else:
                print(f"  ✗ Score file not found for {category}")
                results[category] = {"accuracy": 0.0}
        
        return results
    
    def generate_analysis(self, results: Dict[str, Dict[str, float]]):
        """Generate comprehensive analysis and visualizations."""
        print(f"\n=== Generating analysis for {self.model_name} ===")
        
        # Create analysis directory
        analysis_dir = Path("tool_scaling_analysis")
        analysis_dir.mkdir(exist_ok=True)
        
        # Parse results into structured format
        data = []
        for category, metrics in results.items():
            parts = category.split('_')
            tool_count = int(parts[2])
            position = int(parts[4])
            accuracy = metrics['accuracy']
            
            data.append({
                'category': category,
                'tool_count': tool_count,
                'solution_position': position,
                'accuracy': accuracy
            })
        
        df = pd.DataFrame(data)
        
        # Save raw data
        df.to_csv(analysis_dir / f"{self.model_name}_tool_scaling_results.csv", index=False)
        
        # Generate visualizations
        self._create_visualizations(df, analysis_dir)
        
        # Generate text report
        self._create_text_report(df, analysis_dir)
        
        print(f"  ✓ Analysis saved to {analysis_dir}/")
    
    def _create_visualizations(self, df: pd.DataFrame, output_dir: Path):
        """Create visualization plots."""
        plt.style.use('default')
        
        # 1. Accuracy vs Tool Count (by position)
        plt.figure(figsize=(12, 8))
        for pos in sorted(df['solution_position'].unique()):
            pos_data = df[df['solution_position'] == pos]
            plt.plot(pos_data['tool_count'], pos_data['accuracy'], 
                    marker='o', linewidth=2, markersize=8, 
                    label=f'Position {pos}')
        
        plt.xlabel('Number of Tools', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Tool Scaling Performance: {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(output_dir / f"{self.model_name}_accuracy_vs_tool_count.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmap of accuracy by tool count and position
        pivot_df = df.pivot(index='solution_position', columns='tool_count', values='accuracy')
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
        plt.title(f'Accuracy Heatmap: {self.model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Tools', fontsize=12)
        plt.ylabel('Solution Tool Position', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"{self.model_name}_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Position effect analysis
        plt.figure(figsize=(10, 6))
        position_means = df.groupby('solution_position')['accuracy'].mean()
        plt.bar(position_means.index, position_means.values, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.xlabel('Solution Tool Position', fontsize=12)
        plt.ylabel('Average Accuracy', fontsize=12)
        plt.title(f'Average Accuracy by Solution Position: {self.model_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(position_means.values):
            plt.text(position_means.index[i], v + 0.01, f'{v:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{self.model_name}_position_effect.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Created visualizations")
    
    def _create_text_report(self, df: pd.DataFrame, output_dir: Path):
        """Create detailed text analysis report."""
        report_file = output_dir / f"{self.model_name}_analysis_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Tool Scaling Analysis Report: {self.model_name}\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes how model performance changes with:\n")
            f.write("1. **Tool Count**: Number of available tools (1, 2, 5, 10, 20, 40, 80)\n")
            f.write("2. **Solution Position**: Position of the correct tool in the list\n\n")
            
            # Overall statistics
            f.write("## Overall Performance\n\n")
            f.write(f"- **Average Accuracy**: {df['accuracy'].mean():.3f}\n")
            f.write(f"- **Best Performance**: {df['accuracy'].max():.3f}\n")
            f.write(f"- **Worst Performance**: {df['accuracy'].min():.3f}\n")
            f.write(f"- **Standard Deviation**: {df['accuracy'].std():.3f}\n\n")
            
            # Tool count analysis
            f.write("## Tool Count Analysis\n\n")
            tool_count_stats = df.groupby('tool_count')['accuracy'].agg(['mean', 'std', 'min', 'max'])
            f.write("| Tool Count | Mean Accuracy | Std Dev | Min | Max |\n")
            f.write("|------------|---------------|---------|-----|-----|\n")
            for count, stats in tool_count_stats.iterrows():
                f.write(f"| {count} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |\n")
            f.write("\n")
            
            # Position analysis
            f.write("## Position Effect Analysis\n\n")
            position_stats = df.groupby('solution_position')['accuracy'].agg(['mean', 'std', 'min', 'max'])
            f.write("| Position | Mean Accuracy | Std Dev | Min | Max |\n")
            f.write("|----------|---------------|---------|-----|-----|\n")
            for pos, stats in position_stats.iterrows():
                f.write(f"| {pos} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |\n")
            f.write("\n")
            
            # Detailed results table
            f.write("## Detailed Results\n\n")
            f.write("| Tool Count | Position | Accuracy | Category |\n")
            f.write("|------------|----------|----------|----------|\n")
            for _, row in df.sort_values(['tool_count', 'solution_position']).iterrows():
                f.write(f"| {row['tool_count']} | {row['solution_position']} | {row['accuracy']:.3f} | {row['category']} |\n")
            f.write("\n")
            
            # Key insights
            f.write("## Key Insights\n\n")
            
            # Tool count effect
            first_accuracy = df[df['tool_count'] == 1]['accuracy'].iloc[0] if len(df[df['tool_count'] == 1]) > 0 else 0
            last_accuracy = df[df['tool_count'] == 80]['accuracy'].mean() if len(df[df['tool_count'] == 80]) > 0 else 0
            
            f.write(f"1. **Tool Count Impact**: ")
            if first_accuracy > last_accuracy:
                f.write(f"Performance decreases from {first_accuracy:.3f} (1 tool) to {last_accuracy:.3f} (80 tools average)\n")
            else:
                f.write(f"Performance increases from {first_accuracy:.3f} (1 tool) to {last_accuracy:.3f} (80 tools average)\n")
            
            # Position effect
            pos1_accuracy = df[df['solution_position'] == 1]['accuracy'].mean()
            pos_last = df['solution_position'].max()
            pos_last_accuracy = df[df['solution_position'] == pos_last]['accuracy'].mean()
            
            f.write(f"2. **Position Effect**: ")
            if pos1_accuracy > pos_last_accuracy:
                f.write(f"Performance is better when solution is at position 1 ({pos1_accuracy:.3f}) vs position {pos_last} ({pos_last_accuracy:.3f})\n")
            else:
                f.write(f"Position has minimal impact on performance\n")
            
            # Best/worst configurations
            best_config = df.loc[df['accuracy'].idxmax()]
            worst_config = df.loc[df['accuracy'].idxmin()]
            
            f.write(f"3. **Best Configuration**: {best_config['tool_count']} tools, position {best_config['solution_position']} (accuracy: {best_config['accuracy']:.3f})\n")
            f.write(f"4. **Worst Configuration**: {worst_config['tool_count']} tools, position {worst_config['solution_position']} (accuracy: {worst_config['accuracy']:.3f})\n")
        
        print(f"  ✓ Created analysis report")
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline."""
        print(f"Starting tool scaling evaluation for {self.model_name}")
        print(f"Test configurations: {len(self.test_categories)}")
        
        # Step 1: Generate responses
        self.run_generation()
        
        # Step 2: Evaluate responses
        self.run_evaluation()
        
        # Step 3: Collect and analyze results
        results = self.collect_results()
        self.generate_analysis(results)
        
        print(f"\n=== Tool Scaling Evaluation Complete ===")
        print(f"Results and analysis saved for {self.model_name}")


def main():
    parser = argparse.ArgumentParser(description="Run tool scaling evaluation for BFCL")
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run quick test with subset of configurations")
    parser.add_argument("--generation-only", action="store_true",
                       help="Only run response generation")
    parser.add_argument("--evaluation-only", action="store_true", 
                       help="Only run evaluation (assumes responses exist)")
    parser.add_argument("--analysis-only", action="store_true",
                       help="Only run analysis (assumes evaluations exist)")
    
    args = parser.parse_args()
    
    runner = ToolScalingEvaluationRunner(args.model, args.quick_test)
    
    if args.generation_only:
        runner.run_generation()
    elif args.evaluation_only:
        runner.run_evaluation()
    elif args.analysis_only:
        results = runner.collect_results()
        runner.generate_analysis(results)
    else:
        runner.run_complete_evaluation()


if __name__ == "__main__":
    main()