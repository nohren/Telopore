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

Transformer:
We retrained zhihan1996/DNA\_bert\_6 on CHM13 simulated nanopore data and tested it against CN1 simulated nanopore data and saw roughly 30\% accuracy in terms of chromosome prediction and 70\% accuracy in terms of telomere, subtelomere or non telomere. Even though there was only 30\% chromosome prediction accuracy overall, there was 68\% accuracy on the sub-telomeric regions, 18\% on the telomeric regions and 2\% on non telomeric regions.  In order to really signal focus onto the biology we would need to normalize the sequence lengths in the future.  Currently sub telomeric sequences average 13kb and the others average 4kb. The model is seeing around 3x more sub-telomere chunks than the others. There is a possibility it learned sub telomeric representations of chromosomes over the others.  

Since BERT models typically can’t handle more than 512 tokens in one pass, each sequence is chunked into a 512bp sub-sequence and then a sliding window is passed over to create 507 tokens of 6-mers per sub-sequence. $512 - 6 + 1 = 507$.  All sub-sequences train against their respective sequence label and this is how the model's loss is scored.  This model was trained on four Nvidia H200's over 2 hours. On inference we again split each sequence into respective sub-sequences and then average their logits.  We make sequence label predictions based on the averaged logits of all the sub-sequences in the sequence.  This differs from training where loss is calculated based on the amount of sub-sequences classified correctly over the entire training set (irrespective of sequences). For future training, we can improve upon this and try driving the network towards sequence prediction rather than sub-sequence prediction. 

To avoid chunking a new architecture is needed such as bigbird and longformer which can take in 10 - 50kb as one token. For anyone undertaking this in the future it might be good to also run telogator through or some other tool to verify accuracy of nanosim simulated dataset.  Definitely normalize your Nanosim output sequence length for the three regions to let the biology speak. We can then better answer the question of what kind of data telomeric, sub telomeric, or non telomeric better predicts the chromosome and can better drive the network towards the goal.

model download link which you can use with inference.py
https://drive.google.com/uc?export=download&id=17xNnHWqdKvy2pAI_mkVF_qfSaQvBNcKv

## Slides
https://docs.google.com/presentation/d/1h6-Foln7q_qHxyBhF11Qq3hanGS4dr5PLo51-a4265s/edit?usp=sharing
