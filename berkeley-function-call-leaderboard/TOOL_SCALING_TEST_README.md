# Tool Scaling Test for BFCL

This directory contains a comprehensive test framework for evaluating how Large Language Models (LLMs) perform when presented with different numbers of tools, and how the position of the correct tool affects their performance.

## Overview

The Tool Scaling Test addresses a critical question in function calling evaluation: **How does model performance change as the number of available tools increases, and does the position of the correct tool matter?**

### Key Features

- **Multiple Tool Counts**: Tests with 1, 2, 5, 10, 20, 40, and 80 tools
- **Position Variation**: Places the solution tool at positions 1, 5, 20, and 50
- **Consistent Distractors**: Uses the same set of distractor tools across test cases for fair comparison
- **Comprehensive Analysis**: Generates detailed reports and visualizations
- **Based on Simple Category**: Uses the well-established BFCL simple test cases as the foundation

## Test Configurations

The framework generates 16 different test configurations:

| Tool Count | Position 1 | Position 5 | Position 20 | Position 50 |
|------------|------------|------------|-------------|-------------|
| 1          | ✓          | -          | -           | -           |
| 2          | ✓          | -          | -           | -           |
| 5          | ✓          | ✓          | -           | -           |
| 10         | ✓          | ✓          | -           | -           |
| 20         | ✓          | ✓          | ✓           | -           |
| 40         | ✓          | ✓          | ✓           | -           |
| 80         | ✓          | ✓          | ✓           | ✓           |

Each configuration contains 50 test cases derived from the original BFCL simple dataset.

## Files in this Framework

### Core Scripts

1. **`tool_scaling_test.py`** - Main test generation and analysis script
2. **`run_tool_scaling_evaluation.py`** - Complete evaluation pipeline runner
3. **`demo_tool_scaling_test.py`** - Standalone demonstration script

### Generated Test Files

- **`BFCL_v3_simple_tools_{count}_pos_{position}.json`** - Test data files
- **`tool_scaling_test_ids.json`** - Configuration for selective evaluation
- **`test_configurations.md`** - Documentation of generated configurations

### Analysis Outputs

- **`{model_name}_tool_scaling_results.csv`** - Raw results data
- **`{model_name}_accuracy_vs_tool_count.png`** - Performance vs tool count plot
- **`{model_name}_accuracy_heatmap.png`** - Heatmap of all configurations
- **`{model_name}_position_effect.png`** - Position effect analysis
- **`{model_name}_analysis_report.md`** - Detailed text analysis

## Quick Start

### 1. Generate Test Files

```bash
python tool_scaling_test.py --action generate --max-test-cases 50
```

This creates test files in the `tool_scaling_tests/` directory.

### 2. Install Test Files

The test files have already been copied to `bfcl_eval/data/` and the category mapping has been updated.

### 3. Run a Quick Test

```bash
# Test with a subset of configurations
python run_tool_scaling_evaluation.py --model gpt-4o --quick-test
```

### 4. Run Complete Evaluation

```bash
# Full evaluation with all configurations
python run_tool_scaling_evaluation.py --model gpt-4o
```

### 5. Analyze Existing Results

```bash
# Generate analysis from existing evaluation results
python run_tool_scaling_evaluation.py --model gpt-4o --analysis-only
```

## Individual Test Execution

You can also run individual test configurations:

```bash
# Generate responses for a specific configuration
bfcl generate --model gpt-4o --test-category simple_tools_10_pos_5

# Evaluate the responses
bfcl evaluate --model gpt-4o --test-category simple_tools_10_pos_5
```

## Understanding the Results

### Expected Patterns

Based on cognitive load theory and empirical observations, we expect:

1. **Tool Count Effect**: Performance should decrease as the number of tools increases due to:
   - Increased cognitive load
   - Higher chance of selecting wrong tools
   - Context length limitations

2. **Position Bias**: Performance may vary based on solution tool position due to:
   - Primacy effect (better performance for early positions)
   - Attention mechanisms in transformers
   - Context processing patterns

### Key Metrics

- **Accuracy**: Percentage of test cases where the model selects the correct tool
- **Position Effect**: Difference in accuracy between different solution positions
- **Scaling Factor**: Rate of performance degradation as tool count increases

## Example Analysis

Here's what a typical analysis might reveal:

```
Tool Count Analysis:
- 1 tool: 95% accuracy (baseline)
- 5 tools: 88% accuracy (-7% from baseline)
- 20 tools: 75% accuracy (-20% from baseline)
- 80 tools: 45% accuracy (-50% from baseline)

Position Effect:
- Position 1: 82% average accuracy
- Position 5: 79% average accuracy (-3%)
- Position 20: 71% average accuracy (-11%)
- Position 50: 62% average accuracy (-20%)
```

## Research Applications

This test framework can be used to study:

1. **Model Comparison**: How different models handle tool scaling
2. **Architecture Analysis**: Whether certain architectures are more robust to tool count
3. **Training Effects**: How training data size/diversity affects tool selection
4. **Prompt Engineering**: Optimal ways to present large tool sets
5. **Context Management**: Strategies for handling many tools within context limits

## Customization

### Modifying Tool Counts

Edit the `tool_counts` list in the scripts:

```python
self.tool_counts = [1, 2, 5, 10, 20, 40, 80, 160]  # Add more counts
```

### Changing Positions

Edit the `solution_positions` list:

```python
self.solution_positions = [1, 3, 5, 10, 25, 50]  # Different positions
```

### Using Different Base Tests

Modify the data loading to use different test categories:

```python
# Use parallel tests instead of simple
self.simple_data_path = "bfcl_eval/data/BFCL_v3_parallel.json"
```

## Technical Details

### Tool Selection Strategy

- **Solution Tool**: Always the original tool from the simple test case
- **Distractor Tools**: Selected consistently from the pool of all unique tools in the simple dataset
- **Ordering**: Deterministic based on tool appearance order in the original dataset
- **Repetition**: If more tools are needed than available, tools are repeated in order

### Test Case Generation

1. Load original simple test cases
2. Extract all unique tools to create distractor pool
3. For each configuration (tool count + position):
   - Create new test case with modified tool list
   - Place solution tool at specified position
   - Fill remaining positions with consistent distractors
   - Update test case ID to reflect configuration

### Evaluation Consistency

- Same question text as original simple tests
- Same expected function call as original tests
- Only the available tool list changes between configurations
- Deterministic tool ordering ensures reproducible results

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages:
   ```bash
   pip install matplotlib seaborn pandas
   ```

2. **BFCL Installation Issues**: Use the demo script for standalone testing:
   ```bash
   python demo_tool_scaling_test.py
   ```

3. **Memory Issues**: Reduce the number of test cases:
   ```bash
   python tool_scaling_test.py --action generate --max-test-cases 20
   ```

### Performance Tips

- Use `--quick-test` for initial testing
- Run evaluations in parallel for multiple models
- Use `--analysis-only` to regenerate reports without re-running evaluations

## Contributing

To extend this framework:

1. Add new analysis metrics in the `ToolScalingEvaluator` class
2. Create additional visualization types
3. Implement different tool selection strategies
4. Add support for other BFCL test categories

## Citation

If you use this tool scaling test framework in your research, please cite:

```bibtex
@misc{tool_scaling_test_2024,
  title={Tool Scaling Test Framework for Function Calling Evaluation},
  author={BFCL Contributors},
  year={2024},
  howpublished={Berkeley Function Calling Leaderboard}
}
```

## License

This framework is released under the same license as the Berkeley Function Calling Leaderboard (Apache 2.0).