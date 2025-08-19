import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Load Speech Dataset
# ----------------------------
speech_dir = 'dataset/speech'
speech_data = []

for file in os.listdir(speech_dir):
    if file.endswith('.wav'):
        parts = file.split('_')
        if len(parts) == 3:
            word = parts[1]
            emotion = parts[2].replace('.wav', '')
            speech_data.append({
                'word': word,
                'emotion': emotion,
                'speech_path': os.path.join(speech_dir, file)
            })

speech_df = pd.DataFrame(speech_data)
speech_df.to_csv('speech_word_dataset.csv', index=False)

# ----------------------------
# 2. Load Text Dataset
# ----------------------------
def load_csvs_from_dir(directory):
    combined_df = pd.DataFrame()
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, file))
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

text_train_df = load_csvs_from_dir("dataset/text/train")
text_val_df = load_csvs_from_dir("dataset/text/validation")
text_test_df = load_csvs_from_dir("dataset/text/test")
text_df = pd.concat([text_train_df, text_val_df, text_test_df], ignore_index=True)

# Clean text data - remove NaN values and ensure all text entries are strings
text_df = text_df.dropna(subset=['text', 'label'])
text_df['text'] = text_df['text'].astype(str)
print(f"Text dataset shape after cleaning: {text_df.shape}")
print(f"Text dataset columns: {text_df.columns.tolist()}")

# ----------------------------
# 3. Encode Labels (Shared)
# ----------------------------
label_encoder = LabelEncoder()
all_labels = pd.concat([speech_df['emotion'], text_df['label']], ignore_index=True)
label_encoder.fit(all_labels)

speech_df['label'] = label_encoder.transform(speech_df['emotion'])
text_df['label'] = label_encoder.transform(text_df['label'])

# ----------------------------
# 4. Tokenizer and BERT Model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

# ----------------------------
# 5. Feature Extraction Utils
# ----------------------------
def extract_mfcc(wav_path, max_len=100):
    y, sr = librosa.load(wav_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T

# ----------------------------
# 6. Separate Datasets for Each Modality
# ----------------------------
class AudioDataset(Dataset):
    def __init__(self, speech_df):
        self.speech_df = speech_df.reset_index(drop=True)

    def __len__(self):
        return len(self.speech_df)

    def __getitem__(self, idx):
        speech_row = self.speech_df.iloc[idx]
        mfcc = extract_mfcc(speech_row['speech_path'])
        label = torch.tensor(speech_row['label'], dtype=torch.long)
        return torch.tensor(mfcc, dtype=torch.float32), label

class TextDataset(Dataset):
    def __init__(self, text_df):
        self.text_df = text_df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.text_df)

    def __getitem__(self, idx):
        text_row = self.text_df.iloc[idx]
        
        # Ensure text is a string and handle any potential issues
        text = str(text_row['text']).strip()
        if not text or text == 'nan':
            text = "empty text"  # fallback for empty texts
            
        text_input = tokenizer(text, return_tensors="pt", 
                              padding="max_length", truncation=True, max_length=32)
        text_input = {k: v.squeeze(0) for k, v in text_input.items()}
        label = torch.tensor(text_row['label'], dtype=torch.long)
        return text_input, label

# ----------------------------
# 7. Collate Functions for Each Modality
# ----------------------------
def audio_collate_fn(batch):
    mfccs, labels = zip(*batch)
    mfccs = nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
    labels = torch.tensor(labels)
    return mfccs, labels

def text_collate_fn(batch):
    text_inputs, labels = zip(*batch)
    input_ids = torch.stack([ti['input_ids'] for ti in text_inputs])
    attention_mask = torch.stack([ti['attention_mask'] for ti in text_inputs])
    text_input = {"input_ids": input_ids, "attention_mask": attention_mask}
    labels = torch.tensor(labels)
    return text_input, labels

# ----------------------------
# 8. Individual Models
# ----------------------------
class AudioLSTMModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, (hidden, _) = self.lstm(x)
        # Use the last output
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim*2]
        output = self.classifier(last_output)
        return output

class TextBERTModel(nn.Module):
    def __init__(self, bert_model, num_classes=6):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, text_input):
        with torch.no_grad():
            bert_output = self.bert(**text_input)
            text_feat = bert_output.last_hidden_state[:, 0, :]  # CLS token
        
        output = self.classifier(text_feat)
        return output

# ----------------------------
# 9. Training Functions
# ----------------------------
def train_individual_model(model, train_loader, val_loader, epochs=15, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            inputs, labels = batch
            labels = labels.to(device)
            
            if isinstance(inputs, dict):  # Text input
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:  # Audio input
                inputs = inputs.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        train_acc = total_correct / total_samples
        train_loss = total_loss / len(train_loader)
        
        # Validation
        val_acc, val_loss = evaluate_individual_model(model, val_loader)
        
        print(f"Epoch {epoch+1}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc*100:.2f}%")
    
    return model

def evaluate_individual_model(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            labels = labels.to(device)
            
            if isinstance(inputs, dict):  # Text input
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:  # Audio input
                inputs = inputs.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss

# ----------------------------
# 10. Train Audio LSTM Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label_encoder.classes_)

# Create audio dataset and data loaders
audio_dataset = AudioDataset(speech_df)
audio_train_indices, audio_val_indices = train_test_split(
    list(range(len(audio_dataset))), test_size=0.2, random_state=42)

audio_train_loader = DataLoader(
    torch.utils.data.Subset(audio_dataset, audio_train_indices),
    batch_size=16, shuffle=True, collate_fn=audio_collate_fn)
audio_val_loader = DataLoader(
    torch.utils.data.Subset(audio_dataset, audio_val_indices),
    batch_size=16, shuffle=False, collate_fn=audio_collate_fn)

# Train Audio LSTM Model
print("Training Audio LSTM Model...")
audio_model = AudioLSTMModel(num_classes=num_classes)
audio_model = train_individual_model(audio_model, audio_train_loader, audio_val_loader)

# ----------------------------
# 11. Train Text BERT Model
# ----------------------------
# Create text dataset and data loaders
text_dataset = TextDataset(text_df)
text_train_indices, text_val_indices = train_test_split(
    list(range(len(text_dataset))), test_size=0.2, random_state=42)

text_train_loader = DataLoader(
    torch.utils.data.Subset(text_dataset, text_train_indices),
    batch_size=16, shuffle=True, collate_fn=text_collate_fn)
text_val_loader = DataLoader(
    torch.utils.data.Subset(text_dataset, text_val_indices),
    batch_size=16, shuffle=False, collate_fn=text_collate_fn)

# Train Text BERT Model
print("\nTraining Text BERT Model...")
text_model = TextBERTModel(bert_model, num_classes=num_classes)
text_model = train_individual_model(text_model, text_train_loader, text_val_loader)

# ----------------------------
# 12. Late Fusion Model
# ----------------------------
class LateFusionModel(nn.Module):
    def __init__(self, audio_model, text_model, fusion_method='average'):
        super().__init__()
        self.audio_model = audio_model
        self.text_model = text_model
        self.fusion_method = fusion_method
        
        if fusion_method == 'weighted':
            self.audio_weight = nn.Parameter(torch.tensor(0.5))
            self.text_weight = nn.Parameter(torch.tensor(0.5))
        elif fusion_method == 'learned':
            # Learn to combine the logits
            self.fusion_layer = nn.Sequential(
                nn.Linear(12, 64),  # 6 classes * 2 modalities
                nn.ReLU(),
                nn.Linear(64, 6)
            )

    def forward(self, audio_input, text_input):
        audio_logits = self.audio_model(audio_input)
        text_logits = self.text_model(text_input)
        
        if self.fusion_method == 'average':
            # Simple average of predictions
            fused_logits = (audio_logits + text_logits) / 2
        elif self.fusion_method == 'weighted':
            # Learnable weighted average
            weights = F.softmax(torch.stack([self.audio_weight, self.text_weight]), dim=0)
            fused_logits = weights[0] * audio_logits + weights[1] * text_logits
        elif self.fusion_method == 'learned':
            # Learn to combine concatenated logits
            combined = torch.cat([audio_logits, text_logits], dim=1)
            fused_logits = self.fusion_layer(combined)
        
        return fused_logits, audio_logits, text_logits

# ----------------------------
# 13. Combined Dataset for Late Fusion Evaluation
# ----------------------------
class CombinedDataset(Dataset):
    def __init__(self, speech_df, text_df):
        self.speech_df = speech_df.reset_index(drop=True)
        self.text_df = text_df.reset_index(drop=True)

    def __len__(self):
        return min(len(self.speech_df), len(self.text_df))

    def __getitem__(self, idx):
        speech_row = self.speech_df.iloc[idx]
        text_row = self.text_df.iloc[idx]

        mfcc = extract_mfcc(speech_row['speech_path'])
        label = torch.tensor(speech_row['label'], dtype=torch.long)

        text_input = tokenizer(text_row['text'], return_tensors="pt", 
                              padding="max_length", truncation=True, max_length=32)
        text_input = {k: v.squeeze(0) for k, v in text_input.items()}

        return torch.tensor(mfcc, dtype=torch.float32), text_input, label

def combined_collate_fn(batch):
    mfccs, text_inputs, labels = zip(*batch)
    
    # Pad MFCC sequences
    mfccs = nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
    
    # Stack BERT inputs
    input_ids = torch.stack([ti['input_ids'] for ti in text_inputs])
    attention_mask = torch.stack([ti['attention_mask'] for ti in text_inputs])
    text_input = {"input_ids": input_ids, "attention_mask": attention_mask}
    
    labels = torch.tensor(labels)
    return mfccs, text_input, labels

# Create combined dataset for evaluation
combined_dataset = CombinedDataset(speech_df, text_df)
combined_train_indices, combined_val_indices = train_test_split(
    list(range(len(combined_dataset))), test_size=0.2, random_state=42)

combined_val_loader = DataLoader(
    torch.utils.data.Subset(combined_dataset, combined_val_indices),
    batch_size=16, shuffle=False, collate_fn=combined_collate_fn)

# ----------------------------
# 14. Evaluate Late Fusion Methods
# ----------------------------
fusion_methods = ['average', 'weighted', 'learned']

for method in fusion_methods:
    print(f"\nEvaluating Late Fusion with {method} method...")
    
    # Create late fusion model
    late_fusion_model = LateFusionModel(audio_model, text_model, fusion_method=method)
    late_fusion_model = late_fusion_model.to(device)
    
    # If using learned fusion, train the fusion layer
    if method == 'learned':
        optimizer = torch.optim.Adam(late_fusion_model.fusion_layer.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train fusion layer for a few epochs
        print("Training fusion layer...")
        for epoch in range(5):
            late_fusion_model.train()
            total_loss = 0
            for mfccs, text_inputs, labels in combined_val_loader:
                mfccs, labels = mfccs.to(device), labels.to(device)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                
                optimizer.zero_grad()
                fused_logits, _, _ = late_fusion_model(mfccs, text_inputs)
                loss = criterion(fused_logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Fusion training epoch {epoch+1}, Loss: {total_loss/len(combined_val_loader):.4f}")
    
    # Evaluate late fusion model
    late_fusion_model.eval()
    total_correct = 0
    total_samples = 0
    audio_correct = 0
    text_correct = 0
    
    with torch.no_grad():
        for mfccs, text_inputs, labels in combined_val_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            fused_logits, audio_logits, text_logits = late_fusion_model(mfccs, text_inputs)
            
            # Predictions
            fused_preds = torch.argmax(fused_logits, dim=1)
            audio_preds = torch.argmax(audio_logits, dim=1)
            text_preds = torch.argmax(text_logits, dim=1)
            
            # Accuracies
            total_correct += (fused_preds == labels).sum().item()
            audio_correct += (audio_preds == labels).sum().item()
            text_correct += (text_preds == labels).sum().item()
            total_samples += labels.size(0)
    
    fusion_acc = total_correct / total_samples
    audio_acc = audio_correct / total_samples
    text_acc = text_correct / total_samples
    
    print(f"Audio Model Accuracy: {audio_acc*100:.2f}%")
    print(f"Text Model Accuracy: {text_acc*100:.2f}%")
    print(f"Late Fusion ({method}) Accuracy: {fusion_acc*100:.2f}%")
    print("-" * 50)