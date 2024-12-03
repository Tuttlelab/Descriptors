import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

def load_all_descriptors():
    print("\n=== Loading Descriptor Files ===")

    adi = pd.read_csv('adi_output.csv')
    sfi = pd.read_csv('sfi_output.csv')
    ffi = pd.read_csv('ffi_output.csv')
    tfi = pd.read_csv('tfi_output.csv')
    vfi = pd.read_csv('vfi_output.csv')

    print("\nColumns in each file:")
    print("ADI columns:", adi.columns.tolist())
    print("SFI columns:", sfi.columns.tolist())
    print("FFI columns:", ffi.columns.tolist())
    print("TFI columns:", tfi.columns.tolist())
    print("VFI columns:", vfi.columns.tolist())

    merged = pd.merge(sfi, ffi, on='Frame')
    merged = pd.merge(merged, tfi, on='Frame')
    merged = pd.merge(merged, vfi, on='Frame')

    print("\nFinal merged columns:", merged.columns.tolist())
    print(f"Total frames: {len(merged)}")
    return merged

def analyze_state_transitions(df):
    """Analyze transitions and prepare for decision tree"""
    # Calculate relative dominance scores
    df['sheet_score'] = df['total_peptides_in_sheets'] * df['avg_sheet_size']
    df['fiber_score'] = df['total_peptides_in_fibers'] * df['avg_fiber_size']
    df['vesicle_score'] = df['total_peptides_in_vesicles'] * df['avg_vesicle_size']
    df['tube_score'] = df['total_peptides_in_tubes'] * df['avg_tube_size']

    scores = ['sheet_score', 'fiber_score', 'vesicle_score', 'tube_score']
    states = ['sheet', 'fiber', 'vesicle', 'tube']
    df['dominant_state'] = pd.DataFrame([df[score] for score in scores], index=states).idxmax()

    return df

def train_state_classifier(df):
    """Train decision tree on structure metrics"""
    features = [
        'total_peptides_in_sheets', 'avg_sheet_size',
        'total_peptides_in_fibers', 'avg_fiber_size',
        'total_peptides_in_vesicles', 'avg_vesicle_size',
        'total_peptides_in_tubes', 'avg_tube_size'
    ]

    X = df[features]
    y = df['dominant_state']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X_train, y_train)

    return clf, features, clf.score(X_test, y_test)

def plot_aggregation_analysis(df, timestamp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])

    # Stacked area plot
    scores = ['sheet_score', 'fiber_score', 'vesicle_score', 'tube_score']
    labels = ['Sheets', 'Fibers', 'Vesicles', 'Tubes']
    df_norm = df[scores].div(df[scores].sum(axis=1), axis=0)
    ax1.stackplot(df['Frame'], [df_norm[score] for score in scores],
                 labels=labels, alpha=0.6)

    ax1.set_ylabel('Relative Abundance')
    ax1.set_title('Structure Evolution')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)

    # State transitions
    unique_states = sorted(df['dominant_state'].unique())
    state_map = {state: idx for idx, state in enumerate(unique_states)}
    ax2.scatter(df['Frame'], df['dominant_state'].map(state_map),
                c='black', alpha=0.5, s=5)

    ax2.set_yticks(range(len(unique_states)))
    ax2.set_yticklabels(unique_states)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Dominant Structure')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'aggregation_evolution_{timestamp}.png', dpi=300, bbox_inches='tight')

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n=== Starting Analysis ===")

    df = load_all_descriptors()
    df = analyze_state_transitions(df)

    # Train and save decision tree
    clf, features, accuracy = train_state_classifier(df)
    print(f"\nDecision Tree Accuracy: {accuracy:.2f}")

    plt.figure(figsize=(15, 10))
    plot_tree(clf, feature_names=features, class_names=sorted(df['dominant_state'].unique()),
              filled=True, rounded=True)
    plt.savefig(f'decision_tree_{timestamp}.png', dpi=600, bbox_inches='tight')

    # Generate evolution plot
    plot_aggregation_analysis(df, timestamp)

    # Save analysis results
    df[['Frame', 'dominant_state']].to_csv(
        f'aggregation_analysis_{timestamp}.csv', index=False)

    print(f"\nResults saved with timestamp {timestamp}:")
    print(f"- Evolution plot: aggregation_evolution_{timestamp}.png")
    print(f"- Decision tree: decision_tree_{timestamp}.png")
    print(f"- Analysis data: aggregation_analysis_{timestamp}.csv")

if __name__ == "__main__":
    main()