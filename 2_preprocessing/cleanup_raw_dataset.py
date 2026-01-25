import numpy as np
import os
import shutil
from pathlib import Path

"""
Clean up raw dataset by removing all files with null values (all zeros).
These represent frames where hand detection failed.
"""

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

raw_dataset_dir = os.path.join(project_root, "dataset", "Raw")
backup_dir = os.path.join(project_root, "dataset", "Raw_backup")

# Check if raw dataset exists
if not os.path.exists(raw_dataset_dir):
    print(f"❌ Raw dataset not found at: {raw_dataset_dir}")
    exit(1)

# Create backup
if os.path.exists(raw_dataset_dir):
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(raw_dataset_dir, backup_dir)
    print(f"✓ Backup created at: {backup_dir}\n")

# Statistics
total_files = 0
removed_files = 0
kept_files = 0
invalid_by_gesture = {}
invalid_by_participant = {}

print("=" * 70)
print("CLEANING UP RAW DATASET - REMOVING NULL VALUES")
print("=" * 70)

# Process all participants and gestures
for participant in sorted(os.listdir(raw_dataset_dir)):
    participant_path = os.path.join(raw_dataset_dir, participant)
    if not os.path.isdir(participant_path):
        continue
    
    invalid_by_participant[participant] = 0
    print(f"\n[{participant.upper()}]")
    
    for gesture in sorted(os.listdir(participant_path)):
        gesture_path = os.path.join(participant_path, gesture)
        if not os.path.isdir(gesture_path):
            continue
        
        invalid_by_gesture[gesture] = 0
        gesture_removed = 0
        
        for file in os.listdir(gesture_path):
            file_path = os.path.join(gesture_path, file)
            total_files += 1
            
            try:
                data = np.loadtxt(file_path)
                
                # Check if all values are zero (null/invalid detection)
                if np.allclose(data, 0):
                    os.remove(file_path)
                    removed_files += 1
                    gesture_removed += 1
                    invalid_by_gesture[gesture] += 1
                    invalid_by_participant[participant] += 1
                else:
                    kept_files += 1
            except Exception as e:
                print(f"  ⚠ Error reading {file}: {e}")
        
        if gesture_removed > 0:
            print(f"  {gesture}: {gesture_removed} files removed")

print("\n" + "=" * 70)
print("CLEANUP STATISTICS")
print("=" * 70)
print(f"Total files processed: {total_files}")
print(f"Files removed (all zeros): {removed_files}")
print(f"Files kept (valid): {kept_files}")
if total_files > 0:
    print(f"Removal rate: {removed_files/total_files*100:.1f}%")

print("\n" + "-" * 70)
print("INVALID SAMPLES PER GESTURE:")
print("-" * 70)
for gesture in sorted(invalid_by_gesture.keys()):
    count = invalid_by_gesture[gesture]
    if count > 0:
        print(f"{gesture}: {count} files removed")

print("\n" + "-" * 70)
print("INVALID SAMPLES PER PARTICIPANT:")
print("-" * 70)
for participant in sorted(invalid_by_participant.keys()):
    count = invalid_by_participant[participant]
    print(f"{participant}: {count} files removed")

print("\n" + "=" * 70)
if removed_files > 0:
    print(f"✓ Cleanup complete! {removed_files} invalid samples removed.")
else:
    print("✓ No invalid samples found. Dataset is clean!")
print(f"✓ Backup preserved at: {backup_dir}")
print("=" * 70)
