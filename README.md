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
