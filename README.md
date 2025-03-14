# Telopore
Using nanopore data to distinguish telomeric regions and their respective chromosomes

## Data pipeline

### Data pre-processing

`python telomere.py <reference genome>`

### Data simulation

Simulating long reads

F.Y.I NanoSim can only be run on Linux

Install
```
conda create -n nanosim_env
conda install -c bioconda nanosim
```

ex. 
``` 
simulator.py genome \
    --ref_g reference_set/CHM13_telomere.fasta \
    --model_prefix nanosim/human_giab_hg002_sub1M_kitv14_dorado_v3.2.1/training \
    --number 5000 \
    --output nanosim/telopore_sim_1_1000 \
    --num_threads 4
```

### Data post processing
`python src/sim_to_csv.py nanosim/telopore_sim_1_1000_aligned_reads.fasta nanosim/telopore_sim_2_1000_aligned_reads.fasta nanosim/telopore_sim_0_1000_aligned_reads.fasta out.csv`

## ML Training
Retrained zhihan1996/DNA_bert_6 on CHM13 simulated nanopore data. Tested it on CN1 simulated nanopore data and saw 30% accuracy on chromosomes and 70% on if it's a telomere, subtelomere or non telomere.  The input sequence is limited by standard BERT architecture so the token length is capped at 512bp. The model splits sequences into 512bp sub-sequences where all sub-sequences train against their respective sequence labels.  Each token is a sliding 6-mer window of the 512bp sequence.  This model was trained on four Nvidia H200's over 2 hours. Inference splits each sequence into sub-sequences and then makes a prediction on their average logits. Whereas training calculates loss based on how many sub-sequences are classified right/wrong in the entire training set. In other words we are not focusing on classifying sequences correctly, only sub-sequences. This is something that can be improved on. There is also bigbird and longformer, transformer architecture that can take in longer token sequences, possibly 10 - 50kpb. Before going that route more intuition is needed from the underlying nanopore data, their error models/rates and the underlying frequencies k-mers.  Also in order to accurately verify the training set we can use telogator to verify the nanopore data is good as we assume it to be.

model download link which you can use with inference.py
#https://drive.google.com/uc?export=download&id=17xNnHWqdKvy2pAI_mkVF_qfSaQvBNcKv

## Slides
https://docs.google.com/presentation/d/1h6-Foln7q_qHxyBhF11Qq3hanGS4dr5PLo51-a4265s/edit?usp=sharing
