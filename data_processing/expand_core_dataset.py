import openai
import os
import json
from dotenv import load_dotenv

def load_model():
    load_dotenv()
    openai.api_key = os.getenv("API_KEY")

def contains_fact_box(text):
    """
    Check if the text contains a fact box marked with {{ }} in the text.
    """
    return "{{" in text and "}}" in text

def prepare_fine_tuning_file(input_file, output_file):
    """
    Prepare the dataset for fine-tuning by combining 'prompt' and 'original_text'.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            entry = json.loads(line.strip())
            # Construct the combined prompt and completion format for fine-tuning
            fine_tuning_entry = {
                "prompt": f"{entry['prompt']} Original: {entry['original_text']}",
                "completion": entry['completion']
            }
            outfile.write(json.dumps(fine_tuning_entry, ensure_ascii=False) + "\n")

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
                continue  # Skip empty lines
            
            buffer += line  # Add the line to the buffer
            
            try:
                # Try to parse the accumulated buffer as JSON
                entry = json.loads(buffer)
                entries.append(entry)
                buffer = ""  # Clear buffer after successful parse
            except json.JSONDecodeError:
                continue

        # If buffer is not empty after the loop, it means the last entry is invalid
        if buffer.strip():
            print(f"Warning: Incomplete JSON detected at the end of file:\n{buffer}")

    return entries

def generate_expansion_data(few_shot_examples, original_text, instruction_text):
    """
    Generate expanded data using few-shot examples and the base instruction.
    """
    
    few_shot_context = "\n\n".join(
        f"Original: {example['original_text']}\nCompletion: {example['completion']}"
        for example in few_shot_examples[:10]
    )

    prompt = (
        few_shot_context
        + "\n\n"
        + "Prompt:\n"
        + instruction_text
        + "\n\n"
        + original_text
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du er en norsk journalist som tilpasser skriftlige "
             "nyhetsartikler til en sammenhengende, muntlig fremstilling med god flyt og naturlig språk. "
             "Den muntlige versjonen skal være utfyllende og formidle samme informasjon som originalteksten, "
             "men være bedre strukturert for lytteren. Behold essensielle detaljer, men fjern unødvendige "
             "gjentakelser og overflødig informasjon. Lytteren har ikke mulighet til å hoppe over eller "
             "gå tilbake, så informasjonen må presenteres i en logisk rekkefølge. Sørg for at overganger "
             "mellom avsnitt og temaer føles naturlige for en lytter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=6000,
        top_p=1.0,
        frequency_penalty=0.2,
        presence_penalty=0.0
    )

    return response.choices[0].message.content.strip()


def expand_dataset():
    """
    Expand the dataset by generating new examples based on the few-shot expansion set.
    """
    few_shot_examples = read_multiline_jsonl(
        '/Users/Beatrix/Documents/UiB/MS_Informasjonsvitenskap/Masteroppgave/master-project-code/data_processing/core_dataset.jsonl'
    )
    
    expansion_inputs = read_multiline_jsonl(
        '/Users/Beatrix/Documents/UiB/MS_Informasjonsvitenskap/Masteroppgave/master-project-code/data_processing/expansion.jsonl'
    )

    expanded_set = []
    for entry in expansion_inputs:
        original_text = entry["original_text"]
        title = entry["title"]
        unique_id = entry["unique_id"]

        include_fact_box = contains_fact_box(original_text)

        instruction_text = (
            "Skriv om denne artikkelen for en muntlig presentasjon med god flyt. "
            "Bruk eksemplene over som inspirasjon. "
            "Fjern underoverskrifter markert med ||. Bruk enkle setninger, "
            "og omskriv sitater til indirekte tale. Behold alle essensielle poenger, "
            "men fjern overflødig informasjon og gjentakelser."
        )

        # Conditional prompt
        if include_fact_box:
            base_prompt += (
                " Inkluder viktige fakta fra faktabokser markert med {{ }} for å gi kontekst, " 
                "men integrer dem naturlig i teksten uten å sitere hele boksen."
            )

        # New completion text
        generated_completion = generate_expansion_data(
            few_shot_examples,
            original_text,
            instruction_text
        )

        # Build the new JSON entry. Title kept separate
        new_entry = {
            "prompt": instruction_text.strip(),
            "unique_id": unique_id,
            "title": title,
            "original_text": original_text,
            "completion": generated_completion.strip()
        }

        expanded_set.append(new_entry)


    with open('/Users/Beatrix/Documents/UiB/MS_Informasjonsvitenskap/Masteroppgave/master-project-code/data_processing/few_shot_expansions', 'w', encoding='utf-8') as outfile:
        for entry in expanded_set:
            outfile.write(json.dumps(entry, ensure_ascii=False, indent=4) + '\n')


if __name__ == "__main__":
    load_model()
    expand_dataset()