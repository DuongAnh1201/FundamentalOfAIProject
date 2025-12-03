import pandas as pd
import numpy as np
import os

def merge_and_split_data(data_folder='Training data', 
                         train_ratio=0.7, 
                         val_ratio=0.15, 
                         test_ratio=0.15,
                         random_state=42):
    """
    Merge all emotion CSV files, shuffle, and split into train/validation/test sets.
    
    Args:
        data_folder: Path to folder containing emotion CSV files
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, val_df, test_df: DataFrames for each split
    """
    
    # Emotion label mapping
    emotion_files = {
        'vit_angry_features.csv': 0,
        'vit_disgusted_features.csv': 1,
        'vit_fearful_features.csv': 2,
        'vit_happy_features.csv': 3,
        'vit_sad_features.csv': 4,
        'vit_surpised_features.csv': 5
    }
    
    print("=" * 60)
    print("Merging and Preparing Training Data")
    print("=" * 60)
    
    all_dataframes = []
    
    # Read and label each emotion file
    for filename, label in emotion_files.items():
        filepath = os.path.join(data_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping...")
            continue
        
        print(f"\nReading {filename}...")
        df = pd.read_csv(filepath)
        
        # Add emotion label
        df['emotion'] = label
        
        # Remove filename column if it exists (we don't need it for training)
        if 'filename' in df.columns:
            df = df.drop(columns=['filename'])
        
        print(f"  - Loaded {len(df)} samples")
        print(f"  - Features: {df.shape[1] - 1} (plus emotion label)")
        
        all_dataframes.append(df)
    
    if not all_dataframes:
        print("Error: No data files found!")
        return None, None, None
    
    # Merge all dataframes
    print("\n" + "-" * 60)
    print("Merging all data...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total samples: {len(merged_df)}")
    print(f"Total features: {merged_df.shape[1] - 1}")
    
    # Check emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = merged_df['emotion'].value_counts().sort_index()
    emotion_names = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised']
    for emotion_id, count in emotion_counts.items():
        print(f"  {emotion_names[emotion_id]}: {count} samples ({count/len(merged_df)*100:.1f}%)")
    
    # Shuffle the data
    print("\n" + "-" * 60)
    print("Shuffling data...")
    merged_df = merged_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print("Data shuffled!")
    
    # Split into train/validation/test
    print("\n" + "-" * 60)
    print(f"Splitting data (Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%)...")
    
    # First split: separate train from (val + test)
    train_df, temp_df = stratified_split(
        merged_df, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state
    )
    
    # Second split: separate val from test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = stratified_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=random_state
    )
    
    print(f"\nSplit complete!")
    print(f"  Training set:   {len(train_df)} samples ({len(train_df)/len(merged_df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df)} samples ({len(val_df)/len(merged_df)*100:.1f}%)")
    print(f"  Test set:       {len(test_df)} samples ({len(test_df)/len(merged_df)*100:.1f}%)")
    
    return merged_df, train_df, val_df, test_df


def stratified_split(df, test_size, random_state=None):
    """
    Stratified train-test split using pandas (maintains class distribution).
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data for test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, test_df: Split DataFrames
    """
    np.random.seed(random_state)
    
    train_indices = []
    test_indices = []
    
    # Group by emotion label to maintain distribution
    for emotion_label, group in df.groupby('emotion'):
        n_samples = len(group)
        n_test = int(n_samples * test_size)
        
        # Shuffle indices for this emotion
        indices = group.index.tolist()
        np.random.shuffle(indices)
        
        # Split indices
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
    
    # Create split dataframes
    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    
    return train_df, test_df


def save_splits_to_folder(train_df, val_df, test_df, output_folder):
    """
    Save train/val/test splits to a specific folder.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_folder: Folder path to save the files
    """
    # Create folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the splits
    train_path = os.path.join(output_folder, 'train_features.csv')
    val_path = os.path.join(output_folder, 'val_features.csv')
    test_path = os.path.join(output_folder, 'test_features.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  ✓ Saved to {output_folder}/")
    print(f"    - train_features.csv ({len(train_df)} samples)")
    print(f"    - val_features.csv ({len(val_df)} samples)")
    print(f"    - test_features.csv ({len(test_df)} samples)")


if __name__ == "__main__":
    # Configuration
    data_folder = 'Training data'
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    num_iterations = 5
    
    print("=" * 70)
    print("Creating Multiple Data Splits for Cross-Validation")
    print("=" * 70)
    print(f"\nWill create {num_iterations} different train/val/test splits")
    print(f"Each split will be saved in a separate folder (iteration_1, iteration_2, ...)")
    print(f"Split ratios: Train {train_ratio*100:.0f}% | Val {val_ratio*100:.0f}% | Test {test_ratio*100:.0f}%\n")
    
    # First, merge all data (only need to do this once)
    print("Step 1: Loading and merging all data files...")
    print("-" * 70)
    
    # Emotion label mapping
    emotion_files = {
        'vit_angry_features.csv': 0,
        'vit_disgusted_features.csv': 1,
        'vit_fearful_features.csv': 2,
        'vit_happy_features.csv': 3,
        'vit_sad_features.csv': 4,
        'vit_surpised_features.csv': 5
    }
    
    all_dataframes = []
    
    # Read and label each emotion file
    for filename, label in emotion_files.items():
        filepath = os.path.join(data_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping...")
            continue
        
        print(f"Reading {filename}...")
        df = pd.read_csv(filepath)
        df['emotion'] = label
        
        # Remove filename column if it exists
        if 'filename' in df.columns:
            df = df.drop(columns=['filename'])
        
        all_dataframes.append(df)
    
    if not all_dataframes:
        print("Error: No data files found!")
        exit(1)
    
    # Merge all dataframes
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✓ Total samples loaded: {len(merged_df)}")
    print(f"✓ Total features: {merged_df.shape[1] - 1}")
    
    # Check emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = merged_df['emotion'].value_counts().sort_index()
    emotion_names = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised']
    for emotion_id, count in emotion_counts.items():
        print(f"  {emotion_names[emotion_id]}: {count} samples ({count/len(merged_df)*100:.1f}%)")
    
    # Create 5 different splits
    print("\n" + "=" * 70)
    print(f"Step 2: Creating {num_iterations} different data splits...")
    print("=" * 70)
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n--- Iteration {iteration}/{num_iterations} ---")
        print(f"Using random seed: {42 + iteration}")
        
        # Shuffle the data with different seed for each iteration
        shuffled_df = merged_df.sample(frac=1, random_state=42 + iteration).reset_index(drop=True)
        
        # Split into train/validation/test using stratified splitting
        # First split: separate train from (val + test)
        train_df, temp_df = stratified_split(
            shuffled_df, 
            test_size=(val_ratio + test_ratio), 
            random_state=42 + iteration
        )
        
        # Second split: separate val from test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = stratified_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=42 + iteration
        )
        
        # Create output folder for this iteration
        output_folder = f'iteration_{iteration}'
        
        # Save splits to folder
        save_splits_to_folder(train_df, val_df, test_df, output_folder)
        
        print(f"  Summary: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    print("\n" + "=" * 70)
    print("All iterations complete!")
    print("=" * 70)
    print(f"\nCreated {num_iterations} folders:")
    for i in range(1, num_iterations + 1):
        print(f"  - iteration_{i}/ (contains train_features.csv, val_features.csv, test_features.csv)")
    print("\nYou can now train models on each iteration separately!")

