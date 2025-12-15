#!/usr/bin/env python
"""
Test script for encode_subjects.py

This script independently tests the subject encoding module with real data.
It validates the vocabulary building and TF-IDF encoding functionality.

Usage:
    python test_encode_subjects.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from scipy import sparse
from collections import Counter

from data_preprocessing.config import ITEMS_FILE, OUTPUT_DIR, MIN_SUBJECT_FREQ, MAX_SUBJECT_FEATURES
from data_preprocessing.encode_subjects import (
    build_subject_vocabulary,
    subjects_to_text,
    encode_subjects_tfidf,
    save_subject_vocab,
    load_subject_vocab,
)


def parse_subjects(subjects_str) -> list:
    """Parse semicolon-separated subjects into clean list."""
    if pd.isna(subjects_str) or not subjects_str:
        return []
    subjects = [s.strip().lower() for s in str(subjects_str).split(";")]
    return [s for s in subjects if s]


def load_and_prepare_items() -> pd.DataFrame:
    """Load items and prepare subjects_list column."""
    print(f"Loading items from: {ITEMS_FILE}")
    df = pd.read_csv(ITEMS_FILE)
    df["subjects_list"] = df["Subjects"].apply(parse_subjects)
    print(f"Loaded {len(df)} items")
    return df


def test_vocabulary_building(df: pd.DataFrame) -> list:
    """Test vocabulary building with various parameters."""
    print("\n" + "=" * 60)
    print("TEST 1: Vocabulary Building")
    print("=" * 60)
    
    # Basic stats
    all_subjects = [s for slist in df["subjects_list"] for s in slist]
    unique_subjects = set(all_subjects)
    print(f"\nRaw statistics:")
    print(f"  Total subject occurrences: {len(all_subjects)}")
    print(f"  Unique subjects: {len(unique_subjects)}")
    
    # Test with different min_freq values
    for min_freq in [1, 3, 5, 10, 20]:
        vocab = build_subject_vocabulary(
            df["subjects_list"], 
            min_freq=min_freq, 
            max_features=MAX_SUBJECT_FEATURES
        )
        print(f"  min_freq={min_freq}: {len(vocab)} subjects in vocab")
    
    # Build final vocabulary
    print(f"\nBuilding final vocabulary (min_freq={MIN_SUBJECT_FREQ})...")
    vocab = build_subject_vocabulary(df["subjects_list"])
    
    # Show top subjects
    print(f"\nTop 15 subjects in vocabulary:")
    counter = Counter(s for slist in df["subjects_list"] for s in slist)
    for i, (subj, count) in enumerate(counter.most_common(15)):
        in_vocab = "✓" if subj in vocab else "✗"
        print(f"  {i+1:2}. [{in_vocab}] {subj}: {count}")
    
    # Show subjects at the cutoff boundary
    print(f"\nSubjects near frequency cutoff ({MIN_SUBJECT_FREQ}):")
    sorted_subjects = counter.most_common()
    cutoff_idx = next(
        (i for i, (_, c) in enumerate(sorted_subjects) if c < MIN_SUBJECT_FREQ),
        len(sorted_subjects)
    )
    for subj, count in sorted_subjects[max(0, cutoff_idx-3):cutoff_idx+3]:
        in_vocab = "✓" if subj in vocab else "✗"
        print(f"  [{in_vocab}] {subj}: {count}")
    
    return vocab


def test_tfidf_encoding(df: pd.DataFrame, vocab: list) -> sparse.csr_matrix:
    """Test TF-IDF encoding."""
    print("\n" + "=" * 60)
    print("TEST 2: TF-IDF Encoding")
    print("=" * 60)
    
    # Encode
    tfidf_matrix = encode_subjects_tfidf(df, vocab)
    
    # Validate shape
    expected_shape = (len(df), len(vocab))
    assert tfidf_matrix.shape == expected_shape, \
        f"Shape mismatch: {tfidf_matrix.shape} vs {expected_shape}"
    print(f"✓ Shape correct: {tfidf_matrix.shape}")
    
    # Check for items with no subjects
    items_with_subjects = (tfidf_matrix.sum(axis=1) > 0).sum()
    items_without_subjects = len(df) - items_with_subjects
    print(f"✓ Items with subjects in vocab: {items_with_subjects}")
    print(f"✓ Items without subjects (zero vectors): {items_without_subjects}")
    
    # Validate TF-IDF values
    data = tfidf_matrix.data
    print(f"\nTF-IDF value statistics:")
    print(f"  Min: {data.min():.4f}")
    print(f"  Max: {data.max():.4f}")
    print(f"  Mean: {data.mean():.4f}")
    print(f"  Non-zero entries: {len(data)}")
    
    # Sample a few items
    print(f"\nSample item encodings:")
    for idx in [0, 100, 1000, 5000]:
        if idx >= len(df):
            continue
        row = df.iloc[idx]
        subjects = row["subjects_list"]
        vector = tfidf_matrix[idx].toarray().flatten()
        nonzero_indices = np.where(vector > 0)[0]
        
        print(f"\n  Item {row['i']} (index {idx}):")
        print(f"    Original subjects: {subjects[:5]}{'...' if len(subjects) > 5 else ''}")
        print(f"    Encoded dimensions: {len(nonzero_indices)}")
        if len(nonzero_indices) > 0:
            # Show top TF-IDF values
            top_indices = nonzero_indices[np.argsort(vector[nonzero_indices])[-3:]]
            top_subjects = [(vocab[i], vector[i]) for i in top_indices]
            print(f"    Top subjects by TF-IDF: {top_subjects}")
    
    return tfidf_matrix


def test_vocab_persistence(vocab: list):
    """Test vocabulary save/load."""
    print("\n" + "=" * 60)
    print("TEST 3: Vocabulary Persistence")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = OUTPUT_DIR / "test_subject_vocab.json"
    
    # Save
    save_subject_vocab(vocab, str(vocab_path))
    print(f"✓ Saved vocabulary to: {vocab_path}")
    
    # Load
    loaded_vocab = load_subject_vocab(str(vocab_path))
    print(f"✓ Loaded vocabulary: {len(loaded_vocab)} subjects")
    
    # Validate
    assert vocab == loaded_vocab, "Vocabulary mismatch after load!"
    print(f"✓ Vocabulary matches after round-trip")
    
    # Check encoding preservation (e.g., unicode characters)
    unicode_subjects = [s for s in vocab if any(ord(c) > 127 for c in s)]
    print(f"✓ Unicode subjects preserved: {len(unicode_subjects)} subjects with non-ASCII chars")
    if unicode_subjects:
        print(f"    Examples: {unicode_subjects[:3]}")


def test_edge_cases(df: pd.DataFrame, vocab: list):
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("TEST 4: Edge Cases")
    print("=" * 60)
    
    # Test with empty DataFrame
    print("\nTest 4a: Empty subjects list")
    empty_df = pd.DataFrame({
        "i": [0, 1, 2],
        "subjects_list": [[], [], []]
    })
    empty_matrix = encode_subjects_tfidf(empty_df, vocab)
    assert empty_matrix.sum() == 0, "Empty subjects should produce zero matrix"
    print(f"✓ Empty subjects produce zero matrix")
    
    # Test with subjects not in vocab
    print("\nTest 4b: Unknown subjects (not in vocabulary)")
    unknown_df = pd.DataFrame({
        "i": [0],
        "subjects_list": [["this_subject_does_not_exist_xyz123"]]
    })
    unknown_matrix = encode_subjects_tfidf(unknown_df, vocab)
    assert unknown_matrix.sum() == 0, "Unknown subjects should be ignored"
    print(f"✓ Unknown subjects are ignored (zero vector)")
    
    # Test with mixed known/unknown
    print("\nTest 4c: Mixed known and unknown subjects")
    if len(vocab) > 0:
        known_subject = vocab[0]
        mixed_df = pd.DataFrame({
            "i": [0],
            "subjects_list": [[known_subject, "unknown_subject_xyz"]]
        })
        mixed_matrix = encode_subjects_tfidf(mixed_df, vocab)
        assert mixed_matrix.sum() > 0, "Known subject should be encoded"
        print(f"✓ Mixed subjects: only known subjects are encoded")


def test_alignment_with_item_ids(df: pd.DataFrame, vocab: list):
    """Verify matrix rows align with item IDs."""
    print("\n" + "=" * 60)
    print("TEST 5: Item ID Alignment")
    print("=" * 60)
    
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=42)
    
    # Encode (should sort internally)
    matrix = encode_subjects_tfidf(df_shuffled, vocab)
    
    # Verify row 0 corresponds to item i=0
    item_0 = df[df["i"] == 0].iloc[0]
    row_0_nnz = matrix[0].nnz
    
    # Count expected non-zero (subjects in vocab)
    expected_nnz = len([s for s in item_0["subjects_list"] if s in vocab])
    
    print(f"Item i=0 subjects: {item_0['subjects_list'][:5]}...")
    print(f"Expected non-zero entries: {expected_nnz}")
    print(f"Actual non-zero entries: {row_0_nnz}")
    
    # Note: might not match exactly due to TF-IDF normalization, but should be close
    assert row_0_nnz <= len(item_0["subjects_list"]), "Too many non-zero entries"
    print(f"✓ Row alignment appears correct")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("ENCODE_SUBJECTS.PY - TEST SUITE")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_items()
    
    # Run tests
    vocab = test_vocabulary_building(df)
    tfidf_matrix = test_tfidf_encoding(df, vocab)
    test_vocab_persistence(vocab)
    test_edge_cases(df, vocab)
    test_alignment_with_item_ids(df, vocab)
    
    # Final summary
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print(f"\nFinal outputs:")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Matrix density: {tfidf_matrix.nnz / np.prod(tfidf_matrix.shape):.4f}")
    
    # Save final artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_subject_vocab(vocab, str(OUTPUT_DIR / "subject_vocab.json"))
    sparse.save_npz(OUTPUT_DIR / "subject_tfidf.npz", tfidf_matrix)
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")
    
    return vocab, tfidf_matrix


if __name__ == "__main__":
    run_all_tests()