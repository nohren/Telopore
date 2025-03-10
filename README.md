# Telopore
Using nanopore data to distinguish telomeric regions and their respective chromosomes

Simulating long reads

`cd nanoSim`

ex. 
``` simulator.py genome \
    --ref_g ../CHM13_neg_3000.fasta \
    --model_prefix human_NA12878_DNA_FAB49712_guppy/training \
    --number 100 \
    --coverage 1 \
    --output telopore_sim_neg_3000 \
    --num_threads 4
```
