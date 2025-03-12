import numpy as np
import pandas as pd
import torch
import json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import time
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def one_hot_encode(seq, max_length=None):

    # Define the mapping of nucleotides to indices
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # If max_length is not provided, use the sequence length
    if max_length is None:
        max_length = len(seq)
    else:
        # Truncate or pad sequence as needed
        seq = seq[:max_length].ljust(max_length, 'N')
    
    # Initialize the encoding matrix (4 channels for A, C, G, T)
    encoding = np.zeros((4, max_length), dtype=np.float32)
    
    # Fill in the encoding
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            encoding[mapping[nucleotide], i] = 1.0
    
    return encoding

# Custom Dataset for DNA sequences
class DNASequenceDataset(Dataset):
    def __init__(self, sequences, labels, max_length=None):
        self.sequences = sequences
        self.labels = labels
        
        # Find maximum sequence length if not provided
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        self.max_length = max_length
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Encode sequences
        self.encoded_sequences = [one_hot_encode(seq, max_length) for seq in sequences]
        
        # Convert to tensors - first convert list to a single numpy array to avoid the warning
        self.sequence_tensors = torch.FloatTensor(np.array(self.encoded_sequences))
        self.label_tensors = torch.LongTensor(self.encoded_labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequence_tensors[idx], self.label_tensors[idx]
    
    def get_num_classes(self):
        return len(self.label_encoder.classes_)
    
    def get_class_names(self):
        return self.label_encoder.classes_

class DNASequenceCNN(nn.Module):
    def __init__(self, sequence_length, num_classes):
        super(DNASequenceCNN, self).__init__()
        
        # CNN architecture
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Second convolutional layer
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Third convolutional layer
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Fourth convolutional layer
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        flattened_size = 512 * (sequence_length // (2 * 2 * 2 * 2))
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x

def analyze_data(train_sequences, train_chromosomes, test_sequences=None, test_chromosomes=None):

    print("Dataset Analysis:")
    print(f"Number of training sequences: {len(train_sequences)}")
    
    # Analyze sequence lengths
    train_seq_lengths = [len(seq) for seq in train_sequences]
    print(f"Average training sequence length: {np.mean(train_seq_lengths):.2f}")
    print(f"Min training sequence length: {min(train_seq_lengths)}")
    print(f"Max training sequence length: {max(train_seq_lengths)}")
    
    unique_chromosomes = set(train_chromosomes)
    print(f"Number of unique chromosomes: {len(unique_chromosomes)}")
    print("Chromosome distribution:")
    for chrom in unique_chromosomes:
        count = train_chromosomes.count(chrom)
        print(f"  Chromosome {chrom}: {count} sequences ({count/len(train_chromosomes)*100:.2f}%)")
    
    if test_sequences and test_chromosomes:
        print(f"\nNumber of test sequences: {len(test_sequences)}")
        test_seq_lengths = [len(seq) for seq in test_sequences]
        print(f"Average test sequence length: {np.mean(test_seq_lengths):.2f}")
        
        test_unique_chromosomes = set(test_chromosomes)
        unknown_chromosomes = test_unique_chromosomes - unique_chromosomes
        if unknown_chromosomes:
            print(f"Warning: Test set contains chromosomes not in training set: {unknown_chromosomes}")
            
    return max(train_seq_lengths + ([len(seq) for seq in test_sequences] if test_sequences else []))

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track best model
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    model = model.to(device)
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in tqdm(train_loader, desc="data_loader"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accs, val_accs

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()
    
    print("Training history plot saved as 'training_history.png'")

def evaluate_model(model, test_loader, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    original_preds = label_encoder.inverse_transform(all_preds)
    original_labels = label_encoder.inverse_transform(all_labels)
    
    accuracy = accuracy_score(original_labels, original_preds)
    report = classification_report(original_labels, original_preds)
    
    return accuracy, report, original_preds

def predict_chromosomes_cnn(train_sequences, train_chromosomes, test_sequences, test_chromosomes=None, 
                           batch_size=32, num_epochs=50, learning_rate=0.001,
                           validation_split=0.1, random_seed=42):

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    max_length = analyze_data(train_sequences, train_chromosomes, test_sequences, test_chromosomes)
    print(f"Maximum sequence length: {max_length}")
    
    num_samples = len(train_sequences)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    split = int(np.floor(validation_split * num_samples))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_seq_split = [train_sequences[i] for i in train_indices]
    train_chrom_split = [train_chromosomes[i] for i in train_indices]
    val_seq_split = [train_sequences[i] for i in val_indices]
    val_chrom_split = [train_chromosomes[i] for i in val_indices]
    
    print(f"\nCreating datasets with max length {max_length}...")

    train_dataset = DNASequenceDataset(train_seq_split, train_chrom_split, max_length=max_length)
    val_dataset = DNASequenceDataset(val_seq_split, val_chrom_split, max_length=max_length)

    label_encoder = train_dataset.label_encoder

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    num_classes = train_dataset.get_num_classes()
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_dataset.get_class_names()}")
    
    model = DNASequenceCNN(max_length, num_classes)

    print("\nModel architecture:")
    print(model)

    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate
    )
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    if test_chromosomes:
        test_dataset = DNASequenceDataset(test_sequences, test_chromosomes, max_length=max_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        print("\nEvaluating on test set...")
        test_accuracy, test_report, predictions = evaluate_model(model, test_loader, label_encoder)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Classification Report:\n{test_report}")
        
        model_info = {
            'test_accuracy': test_accuracy,
            'test_report': test_report
        }
    else:
        dummy_labels = ["unknown"] * len(test_sequences)
        test_dataset = DNASequenceDataset(test_sequences, dummy_labels, max_length=max_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        print("\nMaking predictions on test sequences...")
        model.eval()
        all_preds = []
        
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
        
        # Convert encoded predictions back to original labels
        predictions = label_encoder.inverse_transform(all_preds)
        
        model_info = {}
    
    print("\nSaving model and metadata...")
    torch.save(model.state_dict(), 'results/dna_cnn_model.pt')
    
    with open('results/dna_cnn_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('results/dna_cnn_info.pkl', 'wb') as f:
        pickle.dump({
            'max_length': max_length,
            'num_classes': num_classes,
            'model_info': model_info
        }, f)
    
    print("Model saved as 'dna_cnn_model.pt'")
    print("Encoder saved as 'dna_cnn_encoder.pkl'")
    print("Model info saved as 'dna_cnn_info.pkl'")
    
    if len(test_sequences) > 0:
        print("\nSample predictions:")
        for i in range(min(5, len(test_sequences))):
            if test_chromosomes:
                print(f"Sequence: {test_sequences[i][:15]}... | True: {test_chromosomes[i]} | Predicted: {predictions[i]}")
            else:
                print(f"Sequence: {test_sequences[i][:15]}... | Predicted: {predictions[i]}")
    
    return predictions

def predict_new_sequences(new_sequences):
    with open('dna_cnn_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    

    with open('dna_cnn_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    model = DNASequenceCNN(model_info['max_length'], model_info['num_classes'])

    model.load_state_dict(torch.load('dna_cnn_model.pt', map_location=device))
    model = model.to(device)
    model.eval()
    
    dummy_labels = ["unknown"] * len(new_sequences)
    new_dataset = DNASequenceDataset(new_sequences, dummy_labels, max_length=model_info['max_length'])
    new_loader = DataLoader(new_dataset, batch_size=32)
    
    # Make predictions
    all_preds = []
    with torch.no_grad():
        for features, _ in new_loader:
            features = features.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
    
    # Convert encoded predictions back to original labels
    predictions = label_encoder.inverse_transform(all_preds)
    
    return predictions

if __name__ == "__main__":
    df_train = pd.read_csv('dataset/CHM13_2995.csv')
    columns = [df_train[column].tolist() for column in df_train.columns]
    train_chromosomes = columns[1]
    train_sequences = columns[3]

    df_test = pd.read_csv('dataset/CN1_2995.csv')
    columns = [df_test[column].tolist() for column in df_test.columns]
    test_chromosomes = columns[1]
    test_sequences = columns[3]
    
    print("\nTraining Data:")
    predictions = predict_chromosomes_cnn(
        train_sequences, 
        train_chromosomes, 
        test_sequences, 
        test_chromosomes,
        batch_size=64,    
        num_epochs=10,     
        learning_rate=0.001,
        validation_split=0.1
    )
    
    print("\nTest Predictions:")
    pred_output = []
    for seq, true_chrom, pred_chrom in zip(test_sequences, test_chromosomes, predictions):
        output = {'sequence': seq, 'ground_truth': true_chrom, 'predict': pred_chrom}
        pred_output.append(output)
        with open(f'results/CNNprediction.json', 'w') as f:
            json.dump(pred_output, f, indent=4)