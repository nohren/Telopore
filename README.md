# Telopore
Using nanopore data to distinguish telomeric regions and their respective chromosomes

## Data pipeline

### Data pre-processing

`python telopore_data.csv <reference genome>`

### Data simulation

Simulating long reads

`cd nanoSim`

NanoSim can only be run on Linux

ex. 
``` 
simulator.py genome \
    --ref_g ../CHM13_neg_3000.fasta \
    --model_prefix human_NA12878_DNA_FAB49712_guppy/training \
    --number 100 \
    --coverage 1 \
    --output telopore_sim_neg_3000 \
    --num_threads 4
```

### Data post processing
`python sim_to_csv.py sim_pos_aligned.fasta sim_neg_aligned.fasta out.csv`

## ML Training
