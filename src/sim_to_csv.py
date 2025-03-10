#!/usr/bin/env python3

import sys
import csv
# import re

def load_c_map(csv_file):
    code_map = {}
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['Code'].strip()
            chrom_raw = row['Chromosome'].strip()

            # Remove extra quotes or trailing commas if present
            # e.g. '"1,' -> '1'
            chrom_clean = chrom_raw.strip('"').rstrip(',')
            code_map[code] = chrom_clean
    return code_map

c_map = load_c_map('c_map.csv')

def get_chromosome(code_str):
    return c_map.get(code_str, "NA")

def parse_header(header_line):
    return header_line[1:].split("_")[0]

def parse_fasta_and_write_to_csv(fasta_file, is_telomere, csv_writer):
    with open(fasta_file, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines) - 1, 2):
            current_line = lines[i].strip()
            next_line = lines[i+1].strip()
            code = parse_header(current_line)
            chromosome = get_chromosome(code)
            sequence = next_line
            csv_writer.writerow([code, chromosome, is_telomere, sequence])  
        

def main(tl_fasta, sub_tl_fasta, non_tl_fasta, out_csv):
    with open(out_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        # Write a header row (Code, Chromosome, Telomere, Sequence)
        writer.writerow(["Code", "Chromosome", "Telomere", "Sequence"])

        # 1) Parse the 'tl' (telomeric) FASTA
        parse_fasta_and_write_to_csv(tl_fasta, 1, writer)

        # 2) Parse the 'sub_tl' (sub-telomeric) FASTA
        parse_fasta_and_write_to_csv(sub_tl_fasta, 2, writer)

        # 2) Parse the 'non_tl' (non-telomeric) FASTA
        parse_fasta_and_write_to_csv(non_tl_fasta, 0, writer)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <telomeric.fasta> <sub_telomeric.fasta> <non_telomeric.fasta> <output.csv>")
        sys.exit(1)

    tl_fasta = sys.argv[1]
    sub_tl_fasta = sys.argv[2]
    non_tl_fasta = sys.argv[3]
    out_csv = sys.argv[4]
    main(tl_fasta, sub_tl_fasta, non_tl_fasta, out_csv)