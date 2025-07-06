# Tool Scaling Test Configurations

This document describes the test configurations generated for evaluating model performance with different numbers of tools.

## Test Parameters
- Tool counts: [1, 2, 5, 10, 20, 40, 80]
- Solution positions: [1, 5, 20, 50]
- Test cases per configuration: 50
- Total unique tools available: 370

## Generated Test Files

### Solution Tool at Position 1
- 1 tools: `BFCL_v3_simple_tools_1_pos_1.json`
- 2 tools: `BFCL_v3_simple_tools_2_pos_1.json`
- 5 tools: `BFCL_v3_simple_tools_5_pos_1.json`
- 10 tools: `BFCL_v3_simple_tools_10_pos_1.json`
- 20 tools: `BFCL_v3_simple_tools_20_pos_1.json`
- 40 tools: `BFCL_v3_simple_tools_40_pos_1.json`
- 80 tools: `BFCL_v3_simple_tools_80_pos_1.json`

### Solution Tool at Position 5
- 1 tools: *Skipped (position > tool count)*
- 2 tools: *Skipped (position > tool count)*
- 5 tools: `BFCL_v3_simple_tools_5_pos_5.json`
- 10 tools: `BFCL_v3_simple_tools_10_pos_5.json`
- 20 tools: `BFCL_v3_simple_tools_20_pos_5.json`
- 40 tools: `BFCL_v3_simple_tools_40_pos_5.json`
- 80 tools: `BFCL_v3_simple_tools_80_pos_5.json`

### Solution Tool at Position 20
- 1 tools: *Skipped (position > tool count)*
- 2 tools: *Skipped (position > tool count)*
- 5 tools: *Skipped (position > tool count)*
- 10 tools: *Skipped (position > tool count)*
- 20 tools: `BFCL_v3_simple_tools_20_pos_20.json`
- 40 tools: `BFCL_v3_simple_tools_40_pos_20.json`
- 80 tools: `BFCL_v3_simple_tools_80_pos_20.json`

### Solution Tool at Position 50
- 1 tools: *Skipped (position > tool count)*
- 2 tools: *Skipped (position > tool count)*
- 5 tools: *Skipped (position > tool count)*
- 10 tools: *Skipped (position > tool count)*
- 20 tools: *Skipped (position > tool count)*
- 40 tools: *Skipped (position > tool count)*
- 80 tools: `BFCL_v3_simple_tools_80_pos_50.json`

## Usage
To run evaluations on these test files:

```bash
# Generate responses for a specific configuration
bfcl generate --model MODEL_NAME --test-category simple_tools_10_pos_1

# Evaluate the responses
bfcl evaluate --model MODEL_NAME --test-category simple_tools_10_pos_1
```
