import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import os
from pathlib import Path

# === 1. Set folder path ===
# Replace with your folder path containing CSV files
folder_path = r"batch_prediction_results.csv"

# === 2. Define class labels (A–Z) ===
class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# === 3. Get all CSV files in folder ===
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

if not csv_files:
    print(f"No CSV files found in {folder_path}")
    exit()

print(f"Found {len(csv_files)} CSV file(s) in the folder.\n")

# === 4. Process each CSV file ===
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    
    print("=" * 80)
    print(f"Processing: {csv_file}")
    print("=" * 80)
    
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Automatically detect true/pred columns
        if "true_label" in df.columns and "predicted_letter" in df.columns:
            y_true = df["true_label"].astype(str).values
            y_pred = df["predicted_letter"].astype(str).values
        elif "true" in df.columns and "predicted" in df.columns:
            y_true = df["true"].astype(str).values
            y_pred = df["predicted"].astype(str).values
        else:
            print(f"⚠️ Skipping {csv_file}: No 'true_label'/'predicted_letter' or 'true'/'predicted' columns found.\n")
            continue
        
        # Compute metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        
        # Print results
        for i, cls in enumerate(class_names):
            print(f"Class {cls}: Precision = {precision[i]:.3f}, Recall = {recall[i]:.3f}, F1 Score = {f1[i]:.3f}")
        
        print(f"\nOverall Accuracy: {accuracy:.3f}\n")
        
        print("Classification Report:")
        print(classification_report(y_true, y_pred, labels=class_names, target_names=class_names, zero_division=0))
        
        # Save to individual text file
        output_filename = f"metrics_{Path(csv_file).stem}.txt"
        output_path = os.path.join(folder_path, output_filename)
        
        with open(output_path, "w") as f:
            f.write(f"Metrics Report for: {csv_file}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, cls in enumerate(class_names):
                f.write(f"Class {cls}: Precision = {precision[i]:.3f}, Recall = {recall[i]:.3f}, F1 Score = {f1[i]:.3f}\n")
            
            f.write(f"\nOverall Accuracy: {accuracy:.3f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred, labels=class_names, target_names=class_names, zero_division=0))
        
        print(f"✅ Metrics saved as '{output_filename}'\n")
        
    except Exception as e:
        print(f"❌ Error processing {csv_file}: {str(e)}\n")
        continue

print("=" * 80)
print("All files processed!")
print("=" * 80)
