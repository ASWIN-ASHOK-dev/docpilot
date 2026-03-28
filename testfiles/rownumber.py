import csv

def add_line_numbers(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t')
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            
            for i, row in enumerate(reader, start=1):
                # Prepend the line number (i) to the existing row list
                writer.writerow([i] + row)

# Usage
add_line_numbers('reviews.tsv', 'reviews_numbered.tsv')