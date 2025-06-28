import json
from pathlib import Path
import os
import json
import subprocess
from datetime import datetime
import random
import argparse

def int_or_str(x: str):
    try:
        return int(x)
    except ValueError:
        return x

parser = argparse.ArgumentParser()

parser.add_argument(
    '--ground-truth-pos',
    nargs='+',
    type=int_or_str,
    required=True,
    help='The position you want the ground truth function to be in relative to other functions'
)

parser.add_argument(
    '--batch-size',
    type=int,
    required=True,
    help="How many functions you want the model to have access to per query"
)

parser.add_argument(
    '--model-name',
    nargs='+',
    required=True,
    help='The model you want to use for the benchmark'
)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
RESULT_FOLDER_PATH = '/Users/bardia/gorilla/berkeley-function-call-leaderboard/result'
SCORE_FOLDER_PATH = '/Users/bardia/gorilla/berkeley-function-call-leaderboard/score'
TEMPLATE_PATH = '/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_static_template.json'
TODAYS_DATE = datetime.today().strftime('%Y-%m-%d')

positions = args.ground_truth_pos
available_models = args.model_name
# 'claude-3-7-sonnet-20250219-FC',
# 'o3-2025-04-16',
# 'DeepSeek-R1-0528-FC',
# 'qwq-32b-FC' : 'sk-a5bb65a48c27445ea2c05e9ae4a599f8', REMOVE THESE MODELS FROM FUTURE RUNS,
# 'nova-pro-v1.0'
# 'gemini-2.5-pro-preview-05-06-FC' : ['bfcl-model-eval', 'us-central1'] REMOVE THESE MODELS FROM FUTURE RUNS

dataset_paths = [
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_irrelevance.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_java.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_javascript.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_live_irrelevance.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_live_multiple.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_live_parallel_multiple.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_live_parallel.json'),
    Path('berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_live_relevance.json'),
    Path('berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_live_simple.json'),
    # Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_multi_turn_base.json'), NO MULTI TURNS FOR NOW
    # Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_multi_turn_long_context.json'),
    # Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_multi_turn_miss_func.json'),
    # Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_multi_turn_miss_param.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_multiple.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_parallel_multiple.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_parallel.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_static_template.json'),
]

answer_paths = [
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_java.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_javascript.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_live_multiple.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_live_parallel_multiple.json'),
    Path('berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_live_parallel.json'),
    Path('berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_live_simple.json'), #COULD POSSIBLE BE AN ISSUE FOR DUPLICATES. COMMENT OUT IN CASE
    #Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_multi_turn_base.json'), NO MULTI TURNS FOR NOW
    #Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_multi_turn_long_context.json'),
    #Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_multi_turn_miss_func.json'),
    #Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_multi_turn_miss_param.json'),
    Path('berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_multiple.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_parallel_multiple.json'),
    Path('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_parallel.json'),
    Path('berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_simple.json')
]

#BELOW IS (GROUND_TRUTH) OF THE POSSIBLE TOOLS
t_star = {}
for path in answer_paths:
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            qid = data["id"]
            tool_obj = data["ground_truth"][0]  # first (and likely only) tool dict
            tool_name = list(tool_obj.keys())[0]  # extract the function name
            t_star[qid] = tool_name
#ENDS HERE FOR COMPILATION CODE

#BELOW IS (ALL) OF THE POSSIBLE TOOLS
unique_tools = []
seen = set()
for path in dataset_paths:
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            for func in entry.get('function', []):
                key = json.dumps(func, sort_keys=True)  # hashable & deduplicatable
                if key not in seen:
                    seen.add(key)
                    unique_tools.append(func)
#ENDS HERE FOR COMPILATION CODE

tool_dict = { t["name"]: t for t in unique_tools }
tool_list = list(tool_dict.values())

def noisy_benchmark(size=128, true_position='middle'):
    random.seed(42069)
    lines = load_template()
    for line in lines:
        id = line['id']
        ground_truth_name = t_star[f'{id}']
        
        for dic in tool_list:
            name = dic['name']
            if name == ground_truth_name or name.endswith(f".{ground_truth_name}"):
                ground_truth_func = dic
        
        try:
            ground_truth_func
        except NameError:
            raise RuntimeError(
                f"No tool found for ground truth '{ground_truth_name}' in ID {id}"
            )

        pool = [f for f in tool_list if f['name'] != ground_truth_name]
        
        sampled_tools = random.sample(pool, size - 1)
        
        if true_position == 'middle':
            pos = size // 2
        elif type(true_position) is int:
            pos = true_position
        elif true_position == 'beginning':
            pos = 0
        elif true_position == 'end':
            pos = -1
        elif true_position == 'random':
            pos = random.randint(0, size - 1)
        
        sampled_tools.insert(pos, ground_truth_func)
    
        seen = set(); deduped=[]
        for f in sampled_tools:
            if f["name"] not in seen:
                seen.add(f["name"])
                deduped.append(f)
        line["function"] = deduped

    return lines
    
def load_template(path='/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_static_template.json'):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def run_generate(model, result_dir, custom_path):
    print(f"\n Generating for {model} → {result_dir}")
    cmd = [
        "python", "-m", "bfcl_eval", "generate",
        "--test-category", "custom",
        "--custom-path", custom_path,
        "--model", model,
        "--result-dir", result_dir,
        "--allow-overwrite",
        "--temperature", "1",
    ]
    if model.startswith('nova'):
        cmd.insert(0, "AWS_SECRET_ACCESS_KEY=mJJQi6nqoRYdA7DL9zqY8fWsZ2/xLwIIq0pvLuvO")
        cmd.insert(0, "AWS_ACCESS_KEY_ID=AKIA3M7ACZMLM4BFXUCI")
    elif model.startswith('gemini'):
        cmd.insert(0, "VERTEX_AI_LOCATION=us-central1")
        cmd.insert(0, "VERTEX_AI_PROJECT_ID=bfcl-model-eval")
    elif model.startswith('o3') or model.startswith('gpt-4o'):
        cmd.insert(0, "OPENAI_API_KEY=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")
    elif model.startswith('DeepSeek'):
        cmd.insert(0, "OPENAI_API_KEY=sk-7bf84cf3889042c9a242f4c00e62b8df")
    elif model.startswith('qwq'):
        cmd.insert(0, "YOUR_DASHSCOPE_API_KEY=sk-a5bb65a48c27445ea2c05e9ae4a599f8")
    elif model.startswith('claude'):
        cmd.insert(0, "ANTHROPIC_API_KEY=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")
    cw_dic = '/Users/bardia/gorilla/berkeley-function-call-leaderboard'
    subprocess.run(" ".join(cmd), shell=True, check=True, cwd=cw_dic)

def run_evaluate(model, result_dir, eval_dir):
    print(f"\n Evaluating {model}")
    cmd = [
        "python", "-m", 'bfcl_eval', "evaluate",
        "--model", model,
        "--result-dir", result_dir,
        "--score-dir", eval_dir
    ]
    cw_dic = '/Users/bardia/gorilla/berkeley-function-call-leaderboard'
    subprocess.run(cmd, check=True, cwd=cw_dic)

def save_benchmark(lines, path):
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

def runner():
    for model in available_models:
        for position in positions:
            try:
                print(f"\n=== Starting run for {model} at position {position} ===")
                random.seed(42069)
                custom_template_path = os.path.join('/Users/bardia/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/', f'BFCL_v3_custom_{position}.json')

                lines = noisy_benchmark(size=BATCH_SIZE, true_position=position)
                save_benchmark(lines, custom_template_path)

                for line in lines:
                    tool_names = [tool['name'] for tool in line['function']]
                    if len(tool_names) != len(set(tool_names)):
                        duplicates = set([name for name in tool_names if tool_names.count(name) > 1])
                        print(f"❗️ Duplicate tools found in ID {line['id']}: {duplicates}")
                        print(f"🔍 Full tool list for {line['id']}: {tool_names}")

                print(f"CREATED NEW BENCHMARK FOR {model}_{BATCH_SIZE}_{position}_{TODAYS_DATE}")
                result_dir = os.path.join(RESULT_FOLDER_PATH, f"{model}_{BATCH_SIZE}_{position}_{TODAYS_DATE}")
                eval_dir = os.path.join(SCORE_FOLDER_PATH, f"{model}_{BATCH_SIZE}_{position}_{TODAYS_DATE}")

                print(f"GENERATING RESULTS FOR {model}_{BATCH_SIZE}_{position}_{TODAYS_DATE}")
                run_generate(model, result_dir, custom_template_path)

                """ print(f"EVALUATING RESULTS FOR {model}_{BATCH_SIZE}_{position}_{TODAYS_DATE}")
                run_evaluate(model, result_dir, eval_dir) """

                """ if os.path.exists(BENCHMARK_PATH):
                    os.remove(BENCHMARK_PATH)
                    print(f" Deleted old benchmark: {BENCHMARK_PATH}") """

            except subprocess.CalledProcessError as e:
                print(f"❌ Error during subprocess for model={model}, position={position}")
                print(e)
            except Exception as a:
                print(f"❗ Unexpected error for model={model}, position={position}: {a}")

if __name__ == "__main__":
    runner()

    

