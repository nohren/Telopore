from tqdm import tqdm
from Bio import SeqIO
import copy
import fire

def load_fna_records(file_path, max_records=10):
    """
    Generator that yields one record at a time from a large .fna file.
    """
    record_count = 0
    with open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            record_count += 1
            if record_count >= max_records:
                break
            yield record


def main(
        file_path = "data/1_genomic.fna",
        records_num = 2000,
        telomere_length = 3000,
        output_file = "data/CHM13_GCA_3000.fasta",
    ):

    records = []
    for record in tqdm(load_fna_records(file_path, records_num)):
        record_copy = copy.deepcopy(record)
        record.seq = record.seq[:telomere_length]
        records.append(record)
        record_copy.seq = record_copy.seq[-telomere_length:]
        records.append(record_copy)
    
    with open(output_file, "w") as out_handle:
        SeqIO.write(records, out_handle, "fasta")

if __name__ == "__main__":
    fire.Fire(main)