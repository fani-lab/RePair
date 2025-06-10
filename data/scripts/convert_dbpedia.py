#!/usr/bin/env python3
import json

input_file = "/home/RePair/corpora/dbpedia/dbpedia_corpus.jsonl"
output_file = "/home/RePair/corpora/dbpedia/dbpedia_corpus_text.jsonl"

print(f"Converting {input_file} to {output_file}")
print("Changing 'contents' field to 'text' field...")

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line_num, line in enumerate(infile, 1):
        try:
            data = json.loads(line.strip())
            # Convert 'contents' to 'text'
            if 'contents' in data:
                data['text'] = data.pop('contents')
            outfile.write(json.dumps(data) + '\n')
            
            if line_num % 100000 == 0:
                print(f"Processed {line_num} lines...")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_num}: {e}")
            continue

print(f"Conversion complete! Output saved to {output_file}")
