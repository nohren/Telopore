import pandas as pd
import fire
from tqdm import tqdm  
import json


def find_repeats_in_sequence(sequence, min_repeat_length=3, min_occurrences=2):
    sequence = sequence.upper()
    
    repeats = {}

    k_list = [min_repeat_length-1, min_repeat_length, min_repeat_length+1]
    
    for k in tqdm(k_list, desc="Finding repeats"):
        # Generate all k-mers
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            
            # Skip if this k-mer has already been counted as part of a longer repeat
            if any(kmer in longer_repeat for longer_repeat in repeats.keys() if len(longer_repeat) > k):
                continue
                
            count = 0
            count = sequence.count(kmer)
            
            if count >= min_occurrences:
                repeats[kmer] = count
    
    sorted_repeats = {k: v for k, v in sorted(repeats.items(), key=lambda item: item[1], reverse=True)}
    
    return sorted_repeats

def analyze_gene_sequence(sequence, min_repeat_length=3, min_occurrences=2, max_results=20):
    # Find repeats
    repeats = find_repeats_in_sequence(sequence, min_repeat_length, min_occurrences)
    
    tandem_repeats = []
    output_repeats = {}
    for repeat in repeats:
        if repeat * 2 in sequence:
            tandem_repeats.append(repeat)

    is_telomere = False
    
    if tandem_repeats:
        print("\nTandem repeats (appearing consecutively):")
        for repeat in tandem_repeats[:10]:  # Show top 10
            if repeat == 'TTAGGG' or repeat == 'AATCCC':
                is_telomere = True
            output_repeats[repeat] = repeats[repeat]
            print(f"{repeat} (occurs {repeats[repeat]} times)")
    
    return output_repeats, is_telomere


def main(
    input_file='dataset/CHM13_2995.csv',
    min_repeat_length=6,
    min_occurrences=3,
    max_results=20,
    output_file='dataset/CHM13_2995_frequency.json',
    ):

    df = pd.read_csv(input_file)

    # Convert to lists
    headers = df.columns.tolist()
    columns = [df[column].tolist() for column in df.columns]
    chromosomelist = columns[1]
    telomerelist = columns[2]
    sequencelist = columns[3]

    check_telo_list = []
    for i, sequence in enumerate(sequencelist):
        output = {'sequence': sequence}
        output['chromosome'] = chromosomelist[i]
        output['gt_telomere'] = telomerelist[i]
        print(f"Analyzing sequence {i+1}...")
        output_repeats, is_telomere = analyze_gene_sequence(sequence, min_repeat_length, min_occurrences, max_results)
        output['predict_telomere'] = is_telomere
        output['repeats'] = output_repeats
        check_telo_list.append(output)

        with open(f'{output_file}', 'w') as f:
            json.dump(check_telo_list, f, indent=4)




if __name__ == "__main__":
    fire.Fire(main)