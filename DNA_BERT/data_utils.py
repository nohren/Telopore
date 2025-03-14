import torch
import pandas as pd
from multitask_dnabert import MultiTaskDNABERT
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer
import multiprocessing

num_cores = multiprocessing.cpu_count()
print("Number of CPU cores:", num_cores)

def chr_sort_key(c):
    # interpret c as an int if possible, put X=23, Y=24, else bigger
    # adjust as you like:
    if str(c).isdigit():
        return int(c)
    elif c == "X":
        return 23
    elif c == "Y":
        return 24
    else:
        return 99999
         
        
def seq_to_kmers(seq, k=6):
    """
    Convert a DNA sequence into overlapping k-mers,
    then join them with spaces for DNABERT's tokenizer.
    """
    seq = seq.upper()
    kmers = []
    for i in range(len(seq) - k + 1):
        kmers.append(seq[i:i+k])
    return " ".join(kmers)

def chunk_sequence(seq, chunk_size=512, overlap=50):
    """
    Return a list of overlapping substrings from `seq`.
    Example: first chunk covers [0:512], next chunk covers [462:974], etc.
    """
    chunks = []
    start = 0
    length = len(seq)
    while start < length:
        end = start + chunk_size
        chunk = seq[start:end]
        chunks.append(chunk)
        if end >= length:
            break
        # move to next chunk with overlap
        start += (chunk_size - overlap)
    return chunks

class DNABertChunkedDataset(Dataset):
    """
    - Each row in the CSV can produce multiple chunk-examples (if the sequence is long).
    - We'll store them as separate items in the dataset.
    """
    def __init__(self, df, tokenizer, k=6, chunk_size=512, overlap=50, max_length=512):
        self.samples = []  # will hold dicts of form: { "kmers_str":..., "chr_label":..., "tel_label":... }
        self.tokenizer = tokenizer
        self.k = k
        self.max_length = max_length

        for idx in range(len(df)):
            row = df.iloc[idx]
            sequence = row["Sequence"]
            chr_label = row["chr_label"]
            tel_label = row["tel_label"]

            # Split the full sequence into overlapping chunks
            seq_chunks = chunk_sequence(sequence, chunk_size=chunk_size, overlap=overlap)
            for ch in seq_chunks:
                if len(ch) < self.k:
                    continue
                # Convert that chunk to k-mer text
                kmers_str = seq_to_kmers(ch, k=self.k)
                self.samples.append({
                    "kmers_str": kmers_str,
                    "chr_label": chr_label,
                    "tel_label": tel_label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        kmers_str = item["kmers_str"]
        chr_label = item["chr_label"]
        tel_label = item["tel_label"]

        # Tokenize the k-mer string
        encoding = self.tokenizer(
            kmers_str,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels_chr': torch.tensor(chr_label, dtype=torch.long),
            'labels_tel': torch.tensor(tel_label, dtype=torch.long)
        }


class DoTrain():
    def __init__(self, data_path, tokenizer_name, model_name, kmer=6):
        self.k = kmer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model_name = model_name
        
        #preprocess
        df = pd.read_csv(data_path)
        # Chromosome strings to integer IDs drop stuff if NaN
        df = df.dropna(subset=["Chromosome"])
        unique_chrs = df["Chromosome"].unique().tolist()
        sorted_chrs = sorted(unique_chrs, key=chr_sort_key)
        
        chr2id = {ch: i for i, ch in enumerate(sorted_chrs)}
        inv_chr2id = {v: k for k, v in chr2id.items()}
        df["chr_label"] = df["Chromosome"].map(chr2id)
        df["tel_label"] = df["Telomere"] # 4.2: Telomere labels are 0/1/2 ints already

        #print tests, should be 0-23
        unique_labels = df["chr_label"].unique()
        print("Unique chromosome labels:", unique_labels)
        print("Min label:", unique_labels.min(), "Max label:", unique_labels.max())
        print("tel unique:", df["tel_label"].unique())
        #print(df["Chromosome"].isnull().sum())  # or df['chr_label'].isnull().sum()
        # print(len(chr2id))
        print("Empty sequences:", (df["Sequence"].isnull() | (df["Sequence"] == "")).sum())
        self.data = df
        self.chr2id = chr2id
        self.inv_chr2id = inv_chr2id
        self.num_chr_labels = len(chr2id)
        self.num_tel_labels = len(df["tel_label"].unique())

        train_ds = DNABertChunkedDataset(
            df=df,
            tokenizer=self.tokenizer,
            k=kmer,
            chunk_size=512,   # chunk char length
            overlap=50,       # overlap in chars
            max_length=512    # DNABERT max token input
        )
        training = int(0.8 * len(train_ds))
        validation = int(0.2 * len(train_ds))
        training, validation = random_split(train_ds, [training, validation])
        train_loader = DataLoader(training, batch_size=16, shuffle=True, num_workers=num_cores-1)
        val_loader = DataLoader(validation, batch_size=16, shuffle=False, num_workers=num_cores-1)    
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = MultiTaskDNABERT(
            model_name=self.model_name,
            num_chr_labels=self.num_chr_labels,
            num_tel_labels=self.num_tel_labels
        )

    def train(self, epochs=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
        self.model.to(device)

        #cosine decay with warmup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        batches_per_epoch = len(self.train_loader)  

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=batches_per_epoch,  # after T_0 steps, it restarts
            T_mult=1,               # no extension of the cycle length each time
            eta_min=0               # the minimum LR at the cosine nadir
        )

        best_acc_chrom = 0
     
        try:
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                for batch_idx, batch in enumerate(self.train_loader):
                    if batch_idx % 200 == 0:
                        print(f"Batch {batch_idx} out of {len(self.train_loader)}")
                    optimizer.zero_grad()

                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels_chr = batch['labels_chr'].to(device)
                    labels_tel = batch['labels_tel'].to(device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels_chr=labels_chr,
                        labels_tel=labels_tel
                    )
                    loss = outputs['loss']
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(self.train_loader)
                print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

                # Validation
                self.model.eval()
                val_loss = 0
                correct_chr, correct_tel = 0, 0
                total_samples = 0
                with torch.no_grad():
                    for batch in self.val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels_chr = batch['labels_chr'].to(device)
                        labels_tel = batch['labels_tel'].to(device)

                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels_chr=labels_chr,
                            labels_tel=labels_tel
                        )
                        loss = outputs['loss']
                        val_loss += loss.item()

                        logits_chr = outputs['logits_chr']
                        logits_tel = outputs['logits_tel']
                        preds_chr = torch.argmax(logits_chr, dim=1)
                        preds_tel = torch.argmax(logits_tel, dim=1)

                        correct_chr += (preds_chr == labels_chr).sum().item()
                        correct_tel += (preds_tel == labels_tel).sum().item()
                        total_samples += len(labels_chr)

                avg_val_loss = val_loss / len(self.val_loader)
                acc_chr = correct_chr / total_samples
                acc_tel = correct_tel / total_samples
                print(f"Val Loss: {avg_val_loss:.4f}, Chr Acc: {acc_chr:.3f}, Tel Acc: {acc_tel:.3f}")
                if acc_chr > best_acc_chrom:
                    best_acc_chrom = acc_chr
                    torch.save(self.model, "model_best.pt")
        except KeyboardInterrupt:
           # print("Training interrupted; saving partial model.")
            #torch.save(model.state_dict(), "my_partial_model.pt")
            print('interrupted')

        print("Training complete.")
        
        

class DoInference:
    def __init__(self, model, data_path, tokenizer_name, kmer=6):
        self.k = kmer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model

        #preprocess
        df = pd.read_csv(data_path)
        # Chromosome strings to integer IDs drop stuff if NaN
        df = df.dropna(subset=["Chromosome"])
        unique_chrs = df["Chromosome"].unique().tolist()
        sorted_chrs = sorted(unique_chrs, key=chr_sort_key)
        
        chr2id = {ch: i for i, ch in enumerate(sorted_chrs)}
        inv_chr2id = {v: k for k, v in chr2id.items()}
        df["chr_label"] = df["Chromosome"].map(chr2id)
        df["tel_label"] = df["Telomere"] # 4.2: Telomere labels are 0/1/2 ints already

        #print tests, should be 0-23
        unique_labels = df["chr_label"].unique()
        print("Unique chromosome labels:", unique_labels)
        print("Min label:", unique_labels.min(), "Max label:", unique_labels.max())
        print("tel unique:", df["tel_label"].unique())
        #print(df["Chromosome"].isnull().sum())  # or df['chr_label'].isnull().sum()
        # print(len(chr2id))
        print("Empty sequences:", (df["Sequence"].isnull() | (df["Sequence"] == "")).sum())
        self.data = df
        self.chr2id = chr2id
        self.inv_chr2id = inv_chr2id

    

        ######################################
        #  G. Chunk-Level -> Sequence-Level Aggregation
        ######################################
        # If each sequence was chunked into N pieces, you might want an overall label for the entire sequence.
        # One approach: gather chunk predictions for the same "Code" or row, then do majority vote or average logits.
        # This is a quick example of how you might do it for "val_ds".

    def predict_sequence(self, model, tokenizer, seq, k=6, chunk_size=512, overlap=50):
        """
        Return average logits across all chunks for (chr, tel).
        """
        seq_chunks = chunk_sequence(seq, chunk_size=chunk_size, overlap=overlap)
        model.eval()

        sum_logits_chr = None
        sum_logits_tel = None
        total_chunks = 0

        with torch.no_grad():
            for ch in seq_chunks:
                if len(ch) < k:
                    continue
                kmers_str = seq_to_kmers(ch, k=k)
                encoding = tokenizer(
                    kmers_str,
                    return_tensors='pt',
                    truncation=True,
                    padding='max_length',
                    max_length=512
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
                logits_chr = outputs['logits_chr']
                logits_tel = outputs['logits_tel']

                if sum_logits_chr is None:
                    sum_logits_chr = logits_chr
                    sum_logits_tel = logits_tel
                else:
                    sum_logits_chr += logits_chr
                    sum_logits_tel += logits_tel
                total_chunks += 1

        avg_chr = sum_logits_chr / total_chunks
        avg_tel = sum_logits_tel / total_chunks
        pred_chr_id = torch.argmax(avg_chr, dim=1).item()
        chr_str = self.inv_chr2id[pred_chr_id]
        pred_tel_id = torch.argmax(avg_tel, dim=1).item()
        return chr_str, pred_tel_id

    def test(self):
        accChrom = []
        accTelo = []
        for index, row in self.data.iterrows():
            sequence = row["Sequence"]
            chrom = row["Chromosome"]
            telo  = row["Telomere"]
            # do something with these values
            chr_str, tel_label = self.predict_sequence(
                    self.model, 
                    self.tokenizer, 
                    seq=sequence,         # your test DNA sequence
                    k=self.k,                  # must match your training k-mer
                    chunk_size=512,       # same chunk size as training
                    overlap=50            # same overlap as training
                )

            # return chr_str, tel_label, chrom, telo
            if chr_str == chrom:
                accChrom.append(1)
            else:
                accChrom.append(0)

            if tel_label == telo:
                accTelo.append(1)
            else:
                accTelo.append(0)

            if index % 50 == 0:
                print(f'{index} predictions made')

        return sum(accChrom)/len(accChrom),sum(accTelo)/len(accTelo)