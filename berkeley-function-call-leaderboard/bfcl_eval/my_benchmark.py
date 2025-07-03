import json
from pathlib import Path
import os
import json
import subprocess
from datetime import datetime
import random
import argparse
import logging
from bfcl_eval.constants.eval_config import (
    DOTENV_PATH,
    PROJECT_ROOT,
    RESULT_PATH,
    SCORE_PATH,
)

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

parser.add_argument(
    '--openai-key',
    required=False,
    help='The api key for OpenAI models'
)

parser.add_argument(
    '--anthropic-key',
    required=False,
    help='The api key for Anthropic models'
)

parser.add_argument(
    '--amazon-key',
    nargs='+',
    required=False,
    help='The api key for Amazon models'
)

parser.add_argument(
    '--deepseek-key',
    required=False,
    help='The api key for DeepSeek models'
)

parser.add_argument(
    '--gemini-key',
    required=False,
    help='The api key for Gemini models'
)

parser.add_argument(
    '--qwen-key',
    required=False,
    help='The api key for Qwen models'
)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
DATA_PATH = os.path.join(PROJECT_ROOT, 'bfcl_eval/data')
TEMPLATE_PATH = os.path.join(DATA_PATH, 'BFCL_v3_static_template.json')
POSSIBLE_ANSWER_PATH = os.path.join(DATA_PATH, 'possible_answer')
TODAYS_DATE = datetime.today().strftime('%Y-%m-%d')

positions = args.ground_truth_pos
available_models = args.model_name

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='my_benchmark.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
console = logging.StreamHandler()
console.setLevel(logging.INFO) 
console.setFormatter(
    logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', 
                      datefmt='%Y-%m-%d %H:%M:%S')
)
logger.addHandler(console)


# 'claude-3-7-sonnet-20250219-FC',
# 'o3-2025-04-16',
# 'DeepSeek-R1-0528-FC',
# 'qwq-32b-FC' REMOVE THESE MODELS FROM FUTURE RUNS,
# 'nova-pro-v1.0'
# 'gemini-2.5-pro-preview-05-06-FC' REMOVE THESE MODELS FROM FUTURE RUNS
# 'gpt-4o-mini-2024-07-18-FC'

dataset_paths = [
    Path(os.path.join(DATA_PATH, 'BFCL_v3_irrelevance.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_java.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_javascript.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_live_irrelevance.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_live_multiple.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_live_parallel_multiple.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_live_parallel.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_live_relevance.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_live_simple.json')),
    # Path(os.path.join(DATA_PATH, 'BFCL_v3_multi_turn_base.json'), NO MULTI TURNS FOR NOW
    # Path(os.path.join(DATA_PATH, 'BFCL_v3_multi_turn_long_context.json'),
    # Path(os.path.join(DATA_PATH, 'BFCL_v3_multi_turn_miss_func.json'),
    # Path(os.path.join(DATA_PATH, 'BFCL_v3_multi_turn_miss_param.json'),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_multiple.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_parallel_multiple.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_parallel.json')),
    Path(os.path.join(DATA_PATH, 'BFCL_v3_static_template.json')),
]

answer_paths = [
    Path(os.path.join(POSSIBLE_ANSWER_PATH, 'BFCL_v3_java.json')),
    Path(os.path.join(POSSIBLE_ANSWER_PATH, 'BFCL_v3_javascript.json')),
    Path(os.path.join(POSSIBLE_ANSWER_PATH, 'BFCL_v3_live_multiple.json')),
    Path(os.path.join(POSSIBLE_ANSWER_PATH, 'BFCL_v3_live_parallel_multiple.json')),
    Path(os.path.join(POSSIBLE_ANSWER_PATH, 'BFCL_v3_live_parallel.json')),
    Path(os.path.join(POSSIBLE_ANSWER_PATH, 'BFCL_v3_live_simple.json')), #COULD POSSIBLE BE AN ISSUE FOR DUPLICATES. COMMENT OUT IN CASE
    #Path(os.path.join(DATA_PATH, 'possible_answer/BFCL_v3_multi_turn_base.json'), NO MULTI TURNS FOR NOW
    #Path(os.path.join(DATA_PATH, 'possible_answer/BFCL_v3_multi_turn_long_context.json'),
    #Path(os.path.join(DATA_PATH, 'possible_answer/BFCL_v3_multi_turn_miss_func.json'),
    #Path(os.path.join(DATA_PATH, 'possible_answer/BFCL_v3_multi_turn_miss_param.json'),
    Path(os.path.join(POSSIBLE_ANSWER_PATH, 'BFCL_v3_multiple.json')),
    Path(os.path.join(POSSIBLE_ANSWER_PATH, 'BFCL_v3_parallel_multiple.json')),
    Path(os.path.join(POSSIBLE_ANSWER_PATH, 'BFCL_v3_parallel.json')),
    Path(os.path.join(POSSIBLE_ANSWER_PATH, 'BFCL_v3_simple.json'))
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
        
        ground_truth_func = None  # start clean
        for dic in tool_list:
            name = dic['name']
            if name == ground_truth_name or name.endswith(f".{ground_truth_name}"):
                ground_truth_func = dic
                break

        """ if not ground_truth_func:
            print(f"❌ Ground truth function '{ground_truth_name}' not found in tool_list for ID: {id}")
            continue

        # DEBUG: Print what tool we're inserting
        print(f"✅ ID {id}: inserting tool '{ground_truth_func['name']}' (ground truth: {ground_truth_name})") """

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
    
def load_template(path=os.path.join(DATA_PATH, 'BFCL_v3_static_template.json')):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def run_generate(model, result_dir, custom_path):
    logger.info(f"\n Generating for {model} → {result_dir}")
    cmd = [
        "python", "-m", "bfcl_eval", "generate",
        "--test-category", "custom",
        "--custom-path", custom_path,
        "--model", model,
        "--result-dir", result_dir,
        "--allow-overwrite",
        "--temperature", "1",
    ]
    env = os.environ.copy()
    if model.startswith('nova'):
        env["AWS_SECRET_ACCESS_KEY"] = args.amazon_key[1]
        env["AWS_ACCESS_KEY_ID"] = args.amazon_key[0]
    elif model.startswith('gemini'):
        env["GEMINI_API_KEY"] = args.gemini_key
    elif model.startswith('o3') or model.startswith('gpt-4o'):
        env["OPENAI_API_KEY"] = args.openai_key
    elif model.startswith('DeepSeek'):
        env["DEEPSEEK_API_KEY"] = args.deepseek_key
    elif model.startswith('qwq'):
        env["DASHSCOPE_API_KEY"] = args.qwen_key
    elif model.startswith('claude'):
        env["ANTHROPIC_API_KEY"] = args.anthropic_key
    
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)

def run_evaluate(model, result_dir, eval_dir, custom_path):
    logger.info("")
    logger.info("Evaluating %s", model)
    try:
        cmd = [
            "python", "-m", 'bfcl_eval', "evaluate",
            "--model", model,
            "--custom-path", custom_path,
            "--test-category", 'custom',
            "--result-dir", result_dir,
            "--score-dir", eval_dir
        ]
        
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    except Exception as e:
        logger.error("Something went wrong during evaluation: %s", e)


def save_benchmark(lines, path):
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

def rename_result(path, model):
    old_filename = 'BFCL_v3_simple_result.json'
    new_filename = 'BFCL_v3_custom_result.json'

    old_filepath = path / model / old_filename
    new_filepath = path / model / new_filename

    try:
        os.rename(old_filepath, new_filepath)
        logger.info("File old_filename=%s successfully changed to new_filename=%s", old_filename, new_filename)
    except Exception as e:
        logger.info("Something went wrong with the file renaming: %s", e)


def runner():
    for model in available_models:
        for position in positions:
            try:
                logger.info("")
                logger.info("=== Starting run for %s at position %s ===", model, position)
                random.seed(42069)
                custom_template_path = os.path.join(DATA_PATH, f'BFCL_v3_custom_{position}.json')

                lines = noisy_benchmark(size=BATCH_SIZE, true_position=position)
                save_benchmark(lines, custom_template_path)

                for line in lines:
                    tool_names = [tool['name'] for tool in line['function']]
                    if len(tool_names) != len(set(tool_names)):
                        duplicates = set([name for name in tool_names if tool_names.count(name) > 1])
                        print(f"❗️ Duplicate tools found in ID {line['id']}: {duplicates}")
                        logger.error(f"❗️ Duplicate tools found in ID {line['id']}: {duplicates}")
                        print(f"🔍 Full tool list for {line['id']}: {tool_names}")

                logger.info("CREATED NEW BENCHMARK FOR %s_%s_%s_%s", model, BATCH_SIZE, position, TODAYS_DATE)
                result_dir = os.path.join(RESULT_PATH, f"{model}_{BATCH_SIZE}_{position}_{TODAYS_DATE}")

                logger.info("RESULT DIRECTORY: %s", result_dir)
                eval_dir = os.path.join(SCORE_PATH, f"{model}_{BATCH_SIZE}_{position}_{TODAYS_DATE}")

                model_dir = Path(result_dir) / model  # model subfolder
                simple_fp = model_dir / "BFCL_v3_simple_result.json"
                custom_fp = model_dir / "BFCL_v3_custom_result.json"

                if not ( simple_fp.exists() or custom_fp.exists() ):
                    # no result file found → we need to generate
                    run_generate(model, result_dir, custom_template_path)
                    rename_result(result_dir, model)
                else:
                    logger.info("Skipping generation for %s at %s: found existing %s", model, position, RESULT_PATH)

                run_evaluate(model, result_dir, eval_dir, custom_template_path)

                """ if os.path.exists(BENCHMARK_PATH):
                    os.remove(BENCHMARK_PATH)
                    print(f" Deleted old benchmark: {BENCHMARK_PATH}") """

            except subprocess.CalledProcessError as e:
                logger.error("❌ Error during subprocess for model=%s, position=%s", model, position)
            except Exception as a:
                logger.error("❗ Unexpected error for model=%s, position=%s: %s", model, position, a)

if __name__ == "__main__":
    runner()

    

