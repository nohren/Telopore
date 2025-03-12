import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Function to create k-mer features from DNA sequences
def create_kmer_features(sequences, k=3, test_sequences=None):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    X_train = vectorizer.fit_transform(sequences)
    
    if test_sequences is not None:
        X_test = vectorizer.transform(test_sequences)
        return X_train, X_test, vectorizer
    
    return X_train, vectorizer

def analyze_data(train_sequences, train_chromosomes, test_sequences=None, test_chromosomes=None):
    print("Dataset Analysis:")
    print(f"Number of training sequences: {len(train_sequences)}")
    

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

def train_and_evaluate_models(X_train, y_train, X_test, y_test):

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:\n{report}")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report
        }
    
    return results

# Function to find the best model with hyperparameter tuning
def optimize_best_model(X_train, y_train, best_model_name):

    if best_model_name == 'Logistic Regression':
        pipeline = Pipeline([
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__penalty': ['l1', 'l2']
        }
        
    elif best_model_name == 'Random Forest':
        pipeline = Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
    elif best_model_name == 'SVM':
        pipeline = Pipeline([
            ('classifier', SVC(random_state=42))
        ])
        
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            'classifier__gamma': ['scale', 'auto', 0.1, 1]
        }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def visualize_results(results):

    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    plt.savefig('results/model_comparison.pdf')
    plt.close()
    
    print("Model comparison visualization saved as 'model_comparison.pdf'")

# Function to run the entire pipeline
def predict_chromosomes(train_sequences, train_chromosomes, test_sequences, test_chromosomes=None, k=5):

    print("Analyzing dataset...")
    analyze_data(train_sequences, train_chromosomes, test_sequences, test_chromosomes)
    
    print("\nCreating k-mer features...")
    X_train, X_test, vectorizer = create_kmer_features(train_sequences, k=k, test_sequences=test_sequences)
    y_train = train_chromosomes
    
    if test_chromosomes:
        print("\nTraining and evaluating models...")
        results = train_and_evaluate_models(X_train, y_train, X_test, test_chromosomes)
        
        # Find the best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
        
        print("\nOptimizing best model with hyperparameter tuning...")
        best_model = optimize_best_model(X_train, y_train, best_model_name)
        
        # Evaluate optimized model
        y_pred = best_model.predict(X_test)
        optimized_accuracy = accuracy_score(test_chromosomes, y_pred)
        optimized_report = classification_report(test_chromosomes, y_pred)
        
        print(f"\nOptimized {best_model_name} Accuracy: {optimized_accuracy:.4f}")
        print(f"Optimized {best_model_name} Classification Report:\n{optimized_report}")
        
        # Add optimized model to results
        results[f'Optimized {best_model_name}'] = {
            'model': best_model,
            'accuracy': optimized_accuracy,
            'report': optimized_report
        }
        
        # Visualize results
        visualize_results(results)
        
        joblib.dump(best_model, 'results/best_chromosome_predictor_model.pkl')
        joblib.dump(vectorizer, 'results/kmer_vectorizer.pkl')
        
        print("\nBest model and vectorizer saved as 'best_chromosome_predictor_model.pkl' and 'kmer_vectorizer.pkl'")
        
        return results, y_pred
    
    else:
        # If no test_chromosomes provided, we're just making predictions
        print("\nTraining model on all data...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("\nMaking predictions on test sequences...")
        predictions = model.predict(X_test)
        
        joblib.dump(model, 'results/chromosome_predictor_model.pkl')
        joblib.dump(vectorizer, 'results/kmer_vectorizer.pkl')
        
        print("\nModel and vectorizer saved as 'chromosome_predictor_model.pkl' and 'kmer_vectorizer.pkl'")
        
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
    
    
    results, predictions = predict_chromosomes(
        train_sequences, 
        train_chromosomes, 
        test_sequences, 
        test_chromosomes,
        k=3  # Use 3-mers (adjust as needed)
    )
    
    print("\nTest Predictions:")
    pred_output = []

    for seq, true_chrom, pred_chrom in zip(test_sequences, test_chromosomes, predictions):
        output = {'sequence': seq, 'ground_truth': true_chrom, 'predict': pred_chrom}
        pred_output.append(output)
        with open(f'results/MLprediction.json', 'w') as f:
            json.dump(pred_output, f, indent=4)


