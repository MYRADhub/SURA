import csv
import re
import argparse

def extract_case_number(case_name):
    """Extracts the numeric part from a case string like 'case_10' -> 10"""
    match = re.search(r'case_(\d+)', case_name)
    return int(match.group(1)) if match else float('inf')

def sort_csv_by_case(input_file, output_file):
    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        if 'Case' not in fieldnames:
            raise ValueError("CSV must contain a 'Case' column.")
        
        rows = list(reader)

    # Sort rows based on the numeric value in the 'Case' field
    rows.sort(key=lambda row: extract_case_number(row['Case']))

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort a CSV file numerically by the 'Case' column.")
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--output', required=True, help='Path to the output (sorted) CSV file')
    args = parser.parse_args()

    input_csv = args.input
    output_csv = args.output
    sort_csv_by_case(input_csv, output_csv)
    print(f"Sorted CSV written to {output_csv}")