import openai
import sys
import os
import json
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

def contains_fact_box(text):
    """
    Check if the text contains a fact box marked with {{ }} in the text.
    """
    return "{{" in text and "}}" in text

def extract_elements(original_text):
    """
    Uses GPT-4o to extract key elements such as names, places, dates, and events.
    """
    prompt = (
        "Analyser følgende nyhetsartikkel og identifiser nøkkelopplysninger. Du skal finne viktige navngitte enheter i teksten:\n\n"
        f"{original_text}\n\n"
        "Oppgi svarene i følgende format:\n"
        "Viktige personer: [Liste av navn]\n"
        "Viktige steder: [Liste av steder]\n"
        "Viktige datoer: [Liste av datoer]\n"
        "Viktige hendelser: [Kort beskrivelse av hendelser]\n"
        "Utfallet av disse hendelsene: [Kort beskrivelse av utfallene]"
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Du er en erfaren journalist som analyserer nyhetsartikler for å identifisere viktige elementer."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000
    )

    extracted_info = response.choices[0].message.content.strip()

    # Print the extracted elements for debugging purposes
    print("\n**Extracted Key Elements:**")
    print(extracted_info)
    print("-" * 60)

    return extracted_info

def generate_prompt(original_text):
    """
    Generate a suitable prompt based on whether the text contains a fact box.
    """
    base_prompt = (
        "Skriv om denne artikkelen for en muntlig presentasjon med god flyt.\n"
        "**Intruksjoner:**\n"
        "- Fjern underoverskrifter markert med `||`.\n"
        "- Skriv med klare og enkle setninger. Unngå unødvendig lange eller kompliserte setninger.\n"
        "- Omskriv sitater til indirekte tale.\n"
        "- Behold essensielle poenger, men fjern overflødig informasjon og gjentakelser.\n"
        "- Unngå bruk av parenteser og forkortelser. Skriv ut forkortelser i sin fulle form.\n" 
        "- Unngå bruk av anførselstegn rundt uttrykk. Omformuler heller for tydelighet.\n"
        "- Sørg for at det høres naturlig ut når det leses høyt.\n"
    )

    if contains_fact_box(original_text):
        base_prompt += (
            "-Inkluder viktige fakta fra faktabokser markert med {{ }} for å gi kontekst. "
            "Integrer dem naturlig i teksten uten å sitere hele boksen."
        )

    return base_prompt

def generate_oral_version(entry, few_shot_examples):
    """
    Uses GPT-4o to rewrite the original text into an oral-friendly version,
    with the dynamically generated prompt.
    """
    entry_prompt = entry.get("prompt", generate_prompt(entry["original_text"]))

    # Extract key elements
    extracted_elements = "\n".join(line.strip() for line in extract_elements(entry["original_text"]).split("\n")).strip()

    # Construct a few-shot prompt using examples from the train dataset
    few_shot_context = "\n\n".join(
        f"Tittel: {example['title']}\n"
        f"{example['original_text']}\n\n"
        f"Muntlig versjon:\n{example['completion']}"
        for example in few_shot_examples
    )

    # Generate rewritten version using few shot context, entry prompt, and extracted elements
    user_prompt = (
        f"Eksemplene nedenfor viser hvordan en nyhetsartikkel omskrives til en muntlig versjon:\n\n"
        f"{few_shot_context}\n\n"
        f"Nå, følg instruksjonene nedenfor og bruk eksemplene over som inspirasjon.\n"
        f"{entry_prompt}\n\n"
        f"Viktige elementer identifisert fra artikkelen:\n{extracted_elements}\n\n"
        f"Integrer denne informajsonen i omskrivingen.\n"
        f"Det er veldig viktig at informasjon knyttet til elementene ikke endres, utelates eller forvrenges.\n\n"
        f"Dette er artikkelen du skal skrive om:\n"
        f"Tittel: {entry['title']}\n"
        f"Artikkel: {entry['original_text']}" 
    )

    # Call OpenAI GPT-4o API
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": 
                "Du er en norsk journalist som tilpasser skriftlige nyhetsartikler til en sammenhengende, muntlig fremstilling med god flyt og naturlig språk. "
                "Den muntlige versjonen skal være utfyllende og formidle samme informasjon som originalteksten, "
                "men være bedre strukturert for lytteren. Behold essensielle detaljer, men fjern unødvendige "
                "gjentakelser og overflødig informasjon. Lytteren har ikke mulighet til å hoppe over eller "
                "gå tilbake, så informasjonen må presenteres i en logisk rekkefølge. Sørg for at overganger "
                "mellom avsnitt og temaer føles naturlige for en lytter."
            },
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7, # A little lower temperature than default for less creativity (default 1). 
        max_tokens=5000
    )

    rewritten_text = response.choices[0].message.content.strip()

    return rewritten_text, entry_prompt, extracted_elements

def process_unseen_articles(few_shot_file, validation_file, output_file):
    """
    Reads all few-shot examples, applies rewriting to unseen validation articles,
    and saves the results in the same format as `written_oral_dataset.jsonl`.
    """
    few_shot_examples = read_multiline_jsonl(few_shot_file)
    validation_articles = read_multiline_jsonl(validation_file)

    results = []
    for entry in validation_articles:
        rewritten_text, entry_prompt, extracted_elements = generate_oral_version(entry, few_shot_examples)

        formatted_entry = {
            "prompt": entry_prompt,
            "extracted_elements": extracted_elements,
            "unique_id": entry["unique_id"],
            "title": entry["title"],
            "original_text": entry["original_text"],
            "completion": rewritten_text 
        }

        results.append(formatted_entry)

    # Save the rewritten articles to a new JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False, indent=4)
            f.write("\n")

    print(f"Rewritten articles saved to {output_file}")

if __name__ == "__main__":
    load_model()

    if len(sys.argv) != 4:
        raise ValueError("Usage: python rewrite_module.py <few_shot_context> <input_file> <output_file>")
    
    FEW_SHOT_CONTEXT = sys.argv[1]
    INPUT_FILE = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]

    process_unseen_articles(FEW_SHOT_CONTEXT, INPUT_FILE, OUTPUT_FILE)