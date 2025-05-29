# spoken-news-adaptation

This project implements a modular rewriting system for adapting written news articles into a format better suited for spoken delivery. It was developed as part of a master's thesis exploring how to improve the listening experience of audio articles by restructuring written articles into more listener-friendly versions.

___

## Disclaimer

Due to the use of private API keys, this repository is intended for **code review and educational inspection only**. Some functions rely on access to OpenAI’s API via a `.env` file and cannot run without valid credentials.

---

## Setup

#### 1. Clone the repository
git clone https://github.com/yourusername/spoken-news-adaptation.git
cd spoken-news-adaptation

#### 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

#### 3. Add `.env` file
Create a file named `.env` in the root directory and add the following line:
OPENAI_API_KEY=your_key_here

---

## Structure

- `rewrite_module.py` – Rewrites articles for spoken delivery using few-shot prompting.
- `correction_module.py` – Removes formatting artifacts, copied quotes, and fact boxes.
- `modification_tracking.py` – Compares original and rewritten articles using SBERT, ROUGE, and TER.
- `hallucination_detection_transparent.py` – Applies rule-based hallucination detection.
- `hallucination_detection_LLM.py` – Uses an LLM for hallucination classification.
- `test_pipeline_main.py` – Runs the full pipeline in sequence.

---

## Run the pipeline

Run all steps in sequence:
python main.py

---

## Note

The `.env` file is required to run any module that calls the OpenAI API. Without it, those parts of the code will not function.
