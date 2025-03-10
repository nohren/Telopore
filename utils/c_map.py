#!/usr/bin/env python3
import csv
import re

seen = {}

def parse_fasta_and_write(fasta_file, writer):
    with open(fasta_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line.startswith('>'):
                line = line.lstrip('>')    
                prefix = line.split('.')[0]
                prefix = prefix.replace('_','-')

                pattern = r'chromosome\s+(\S+)'
                match = re.search(pattern, line, re.IGNORECASE)
                chromosome_str = "N/A"
                if match:
                    chromosome_str = match.group(1).strip('",')  # 
                if prefix in seen:
                    continue
                
                seen[prefix] = chromosome_str
                writer.writerow([prefix,chromosome_str])


def main(pos_fasta, neg_fasta, output_csv):
    with open(output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(["Code", "Chromosome"])
        
        # 1) Parse the "positive" (telomeric) FASTA => isTelomeric=1
        parse_fasta_and_write(pos_fasta, writer)
        
        # 2) Parse the "negative" (non-telomeric) FASTA => isTelomeric=0
        parse_fasta_and_write(neg_fasta, writer)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <telomeric.fasta> <non_telomeric.fasta> <output.csv>")
        sys.exit(1)

    pos_fasta = sys.argv[1]     # e.g. "telomeric.fa"
    neg_fasta = sys.argv[2]     # e.g. "non_telomeric.fa"
    output_csv = sys.argv[3]    # e.g. "combined.csv"

    main(pos_fasta, neg_fasta, output_csv)