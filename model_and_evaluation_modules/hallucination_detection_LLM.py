import sys
import json
import openai
import os
import re
from dotenv import load_dotenv

PROMPT_TEMPLATE = """
Du skal evaluere en omskrevet nyhetsartikkel for faktakonsistens mot den opprinnelige artikkelen.
Målet ditt er å sjekke om det er hallusinasjoner, faktafeil og unøyaktigheter.

Den opprinnelige artikkelen er sannheten. Den omskrevne artikkelen skal bare omformulere, 
men må ikke legge til eller forvrenge fakta. Ikke flagg omformuleringer eller stilistiske endringer.

### Typer hallusinasjoner:
1. Faktisk forvrengning: En faktuell opplysning er endret (f.eks. "5 millioner" til "50 millioner").
2. Fabrikasjon: Ny informasjon er lagt til som ikke eksisterer i originalen.
3. Overdrivelsee eller underdrivelse: Fakta har blitt overdreven eller bagatellisert.
4. Utelatelse: Viktige fakta er fjernet, noe som endrer betydningen.

### Instruksjoner:
Analyser hele den omskrevne artikkelen i forhold til original artikkel, inkludert faktabokser markert med {{}} i teksten. 
Sjekk om det er spesifikke hallusinasjoner, forklar hvorfor de er problematiske, og oppgi hvor sikker du er i vurderingen.

Original arikkel:
{original}

Omskrevet artikkel:
{rewritten}

Returner et gyldig JSON-objekt med følgende format:
```json
{{
    "hallusinasjoner": [
        {{
            "omskrevet_setning": "Den omskrevne setningen her",
            "type_hallusinasjon": "Faktisk forvrengning / Fabrikasjon / Overdrivelse / Utelatelse",
            "forklaring": "Forklaring på hvorfor dette er en hallusinasjon.",
            "modellens_sikkerhet_i_deteksjonen": 1-10
        }}
    ]
}}
"""

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

def load_model():
    load_dotenv()
    openai.api_key = os.getenv("API_KEY")

# Function to clean and extract JSON from LLM response
def extract_json(response_text):
    """Extracts valid JSON from the LLM response, removing Markdown formatting if necessary."""
    # Use regex to extract JSON inside ```json ... ```
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    json_text = match.group(1) if match else response_text.strip()  # Extract JSON or use raw text

    try:
        return json.loads(json_text)  # Convert JSON string to a Python dictionary
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {"error": "Failed to parse LLM response", "raw_response": response_text}

def detect_hallucinations(original_text, rewritten_text):
    """Uses OpenAI's LLM to detect hallucinations in rewritten news articles."""
    prompt = PROMPT_TEMPLATE.format(original=original_text, rewritten=rewritten_text)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Du er en presis faktasjekker som analyserer omskrevne nyhetsartikler"},
                      {"role": "user", "content": prompt}],
            temperature=0.2,  # Low temperature to reduce randomness, focused task
            max_tokens=3000
        )

        return extract_json(response.choices[0].message.content) # Parse json output
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def process_articles(input_file, output_file):
    """Loads articles, runs hallucination detection, and saves results."""
    results = []

    data = read_multiline_jsonl(input_file)

    # Analyse each entry
    for entry in data:
        original_text = entry["original_text"]
        corrected_rewritten_text = entry["corrected_text"]
   
        analysis_result = detect_hallucinations(original_text, corrected_rewritten_text)
        
        if analysis_result:
            results.append({
                "unique_id": entry["unique_id"],
                "title": entry["title"],
                "hallucination_analysis": analysis_result
            })

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Hallucination analysis completed. Results saved to {output_file}")

if __name__ == "__main__":
    load_model()
    
    if len(sys.argv) != 3:
        raise ValueError("Usage: python hallucination_detection_LLM.py <input_file> <output_file>")

    INPUT_FILE = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]
    
    process_articles(INPUT_FILE, OUTPUT_FILE)
    