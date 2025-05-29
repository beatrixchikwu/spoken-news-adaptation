import os
import subprocess

# Define base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
MODULE_DIR = os.path.join(BASE_DIR, "model_and_evaluation_modules")

# Ensure output directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define file paths
few_shot_context = os.path.join(DATASET_DIR, "train_articles.jsonl")
original_articles = os.path.join(DATASET_DIR, "evaluation_articles.jsonl")
rewritten_articles = os.path.join(RESULTS_DIR, "evaluation_rewritten_articles.jsonl")
corrected_articles = os.path.join(RESULTS_DIR, "evaluation_corrected_articles.jsonl")
modification_output = os.path.join(RESULTS_DIR, "evaluation_modification_tracking.jsonl")
transparent_output = os.path.join(RESULTS_DIR, "evaluation_analysis_hallucination_hybrid.jsonl")
llm_output = os.path.join(RESULTS_DIR, "evaluation_analysis_hallucination_LLM1.jsonl")

# Helper to run scripts with arguments
def run_script(script_name, *args):
    script_path = os.path.join(MODULE_DIR, script_name)
    print(f"Running {script_name}...")
    result = subprocess.run(["python", script_path, *args])
    if result.returncode != 0:
        print(f"Error running {script_name}")
    else:
        print(f"Finished {script_name}")

# Pipeline steps
run_script("rewrite_module.py", few_shot_context, original_articles, rewritten_articles)
run_script("correction_module.py", rewritten_articles, corrected_articles)
run_script("modification_tracking.py", corrected_articles, modification_output)
run_script("hallucination_detection_transparent.py", corrected_articles, transparent_output)
run_script("hallucination_detection_LLM.py", corrected_articles, llm_output)

print("All pipeline steps completed.")