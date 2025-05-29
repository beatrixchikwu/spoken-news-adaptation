import re
import sys
import json
import openai
import os
from dotenv import load_dotenv

def load_model():
    load_dotenv()
    openai.api_key = os.getenv("API_KEY")

def read_multiline_jsonl(file_path):
    """
    Reads a JSONL file where entries can span multiple lines.
    Returns a list of parsed JSON objects.
    """
    entries = []
    buffer = ""

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  
            
            buffer += line  
            
            try:
                entry = json.loads(buffer)  
                entries.append(entry)  
                buffer = ""  
            except json.JSONDecodeError:
                continue 

    return entries

def remove_parentheses(rewritten_text):
    """
    Identifies and removes parenthesis in the rewritten text.
    Returns the cleaned text and a list of (full_sentence, parenthetical phrase) tuples.
    """
    detected_parentheses = []

    # Split text into sentences (naive but usually sufficient for news)
    sentences = re.split(r'(?<=[.!?])\s+', rewritten_text)

    for sentence in sentences:
        matches = re.findall(r"\(([^)]+)\)", sentence)
        for match in matches:
            detected_parentheses.append((sentence.strip(), match.strip()))

    # Remove all parentheses from the text
    cleaned_text = re.sub(r"\s*\([^)]*\)", "", rewritten_text).strip()

    return cleaned_text, detected_parentheses

def remove_subheadings(rewritten_text):
    """
    Removes subheadings marked with || ... || in the rewritten text.
    Returns the cleaned text and a list of detected subheadings.
    """
    detected_subheadings = re.findall(r"\|\|.*?\|\|\n?", rewritten_text)

    if detected_subheadings:
        rewritten_text = re.sub(r"\|\|.*?\|\|\n?", "", rewritten_text).strip()
    return rewritten_text, detected_subheadings


def detect_fact_boxes(original_text, rewritten_text):
    """Detects fact boxes that are copied verbatim or if fact box markers {{...}} are still present."""
    issues = []
    fact_boxes = re.findall(r"\{\{(.*?)\}\}", original_text, re.DOTALL)

    for box in fact_boxes:
        if box in rewritten_text:
            match = re.search(rf"([^.]*{re.escape(box)}[^.]*)\.", rewritten_text)
            if match:
                full_sentence = match.group(1) + "."
                issues.append({
                    "error_type": "Fact box content copied verbatim",
                    "fact_box": box,
                    "full_sentence": full_sentence
                })

    # Detect if fact box markers are still in the rewritten text
    fact_box_markers = re.findall(r"\{\{(.*?)\}\}", rewritten_text, re.DOTALL)
    for marker in fact_box_markers:
        issues.append({
            "error_type": "Fact box markers still present",
            "fact_box": marker,
            "full_sentence": f"{{{{ {marker} }}}}"
        })

    return issues


def detect_copied_quotes(original_text, rewritten_text):
    """Detects direct quotes introduced by '–' and checks if they are copied verbatim into the rewritten text."""
    issues = []

    # Match any dash-based quote, with optional preceding punctuation and whitespace
    direct_quotes = re.findall(r"[.!?\n\r]\s*–\s*(.+?)(?=[.!?])", original_text)

    for quote in direct_quotes:
        quote = quote.strip()
        if quote in rewritten_text:
            match = re.search(rf"([^.]*{re.escape(quote)}[^.]*)\.", rewritten_text)
            if match:
                full_sentence = match.group(1).strip() + "."
                issues.append({
                    "error_type": "Direct quote copied verbatim",
                    "original_quote": quote,
                    "full_sentence": full_sentence
                })

    return issues

# Use LLM to correct fact boxes and quotes
def ask_llm_to_fix(issues):
    """Sends all detected issues to LLM in a batch request and retrieves corrected versions."""
    if not issues:
        return {}

    prompt = "Følgende setninger må korrigeres:\n\n"

    for i, issue in enumerate(issues):
        if "fact_box" in issue:
            prompt += f"**Faktaboks {i+1}:**\nOpprinnelig faktaboks: {issue['fact_box']}\nOmgivende setning: {issue['full_sentence']}\n\n"
        else:
            prompt += f"**Direkte sitat {i+1}:**\nOpprinnelig sitat: {issue['original_quote']}\nOmgivende setning: {issue['full_sentence']}\n\n"

    prompt += (
        "Oppgave:\n"
        "- Omskriv faktaboksene slik at de er naturlig integrert i teksten. Inkluder kun viktig fakta fra faktaboksen.\n"
        "- Omskriv sitatene til indirekte tale.\n"
        "- Ikke endre noen andre deler av teksten utenom setningene som må korrigeres.\n"
        "- Behold meningen uendret.\n"
        "- Returner kun de korrigerte setningene i samme rekkefølge.\n"
        "- Ikke legg til ekstra forklaringer, tall, eller nummerering.\n"
        "- Hver korrigert setning skal stå på en egen linje uten nummerering eller ekstra tegn."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Du er en profesjonell korrekturleser og språkvasker på norsk."},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )

        corrected_sentences = response.choices[0].message.content.strip().split("\n")

        # Trim whitespace and remove empty lines
        corrected_sentences = [s.strip() for s in corrected_sentences if s.strip()]

        # Ensure response count matches detected issues
        if len(corrected_sentences) > len(issues):
            print(f"Warning: Expected {len(issues)} corrections, but LLM returned {len(corrected_sentences)}.")
            corrected_sentences = corrected_sentences[:len(issues)]  # Trim extra responses
        elif len(corrected_sentences) < len(issues):
            print(f"Warning: LLM returned fewer corrections than expected ({len(corrected_sentences)} vs {len(issues)}). Some sentences may remain uncorrected.")

        # Ensure proper mapping between issues and corrected sentences
        return {
            issues[i]["full_sentence"]: corrected_sentences[i]
            for i in range(len(corrected_sentences))
        }

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")        
        return {}  

# Apply corrections
def correct_text(original_text, rewritten_text):
    """
    Detects issues in the rewritten text (fact boxes, quotes, subheadings, parentheses),
    removes incorrect elements, and updates the rewritten text.
    """
    # --- 1. Detect and handle subheadings ---
    rewritten_text, removed_subheadings = remove_subheadings(rewritten_text)

    # --- 2. Detect fact boxes and quotes before editing ---
    fact_box_issues = detect_fact_boxes(original_text, rewritten_text)
    quote_issues = detect_copied_quotes(original_text, rewritten_text)
    all_issues = fact_box_issues + quote_issues

    # --- 3. Then remove parentheses ---
    rewritten_text, removed_parentheses = remove_parentheses(rewritten_text)

    # --- 4. Prepare correction log ---
    correction_log = []

    for subheading in removed_subheadings:
        correction_log.append({
            "rewritten_sentence_before_correction": subheading.strip(),
            "corrected_sentence": "(Fjernet feilaktig underoverskrift)"
        })

    for full_sentence, parenthetical in removed_parentheses:
        correction_log.append({
            "rewritten_sentence_before_correction": full_sentence,
            "corrected_sentence": f"(Fjernet parentes ({parenthetical}) for bedre muntlig flyt)"
        })

    # --- 5. Use LLM to fix fact boxes and quotes ---
    corrections = ask_llm_to_fix(all_issues)

    for original, fixed in corrections.items():
        correction_log.append({
            "rewritten_sentence_before_correction": original.strip(),
            "corrected_sentence": fixed.strip()
        })
        rewritten_text = rewritten_text.replace(original, fixed)

    # --- 6. Return result ---
    if not correction_log:
        return rewritten_text, [{"message": "Ingen endringer nødvendig"}]

    return rewritten_text, correction_log

def process_articles(input_file, output_file):
    """
    Processes articles and saves the corrected versions including a list of corrections. 
    Skipping redundant fields if no corrections were needed.
    """
    data = read_multiline_jsonl(input_file)
    
    corrected_articles = []

    for entry in data:
        original_text = entry["original_text"]
        rewritten_text = entry["completion"]  
        corrected_text, correction_log = correct_text(original_text, rewritten_text)

        formatted_entry = {
            "unique_id": entry["unique_id"],
            "title": entry["title"],
            "original_text": original_text
        }

        # If corrections were made, keep both versions
        if correction_log and correction_log[0].get("message") != "Ingen endringer nødvendig":
            formatted_entry["rewritten_article_before_correction"] = rewritten_text
            formatted_entry["corrected_text"] = corrected_text
            formatted_entry["corrections"] = correction_log
        else:
            # No changes, only include corrected_text
            formatted_entry["corrected_text"] = rewritten_text
            formatted_entry["corrections"] = [{"message": "Ingen endringer nødvendig"}]

        corrected_articles.append(formatted_entry)

    # Write results back to JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for result in corrected_articles:
            json.dump(result, f, ensure_ascii=False, indent=4)
            f.write("\n") 

    print(f"Correction process completed. Results saved to {output_file}")


if __name__ == "__main__":
    load_model()

    if len(sys.argv) != 3:
        raise ValueError("Usage: python correction_module.py <input_file> <output_file>")

    INPUT_FILE = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]

    process_articles(INPUT_FILE, OUTPUT_FILE)