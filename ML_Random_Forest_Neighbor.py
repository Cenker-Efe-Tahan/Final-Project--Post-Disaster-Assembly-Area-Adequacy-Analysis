import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

pd.options.mode.chained_assignment = None


def _normalize_mahalle(s: str) -> str:
    if pd.isna(s):
        return ''
    s = str(s).strip()
    # Remove leading ordinal prefixes: "1.", "100.", "2. ", "1.Kadriye" -> "Kadriye"
    s = re.sub(r'^\d+\.?\s*', '', s)
    # Remove common suffixes (order matters — longest first)
    suffixes = [
        r'\bmahallesi\b', r'\bmahalle\b', r'\bmah\b\.?',
        r'\bköyü\b', r'\bköy\b',
    ]
    for pat in suffixes:
        s = re.sub(pat, '', s, flags=re.IGNORECASE)
    # Lowercase BEFORE Turkish char mapping so Ş, İ, etc. are caught
    s = s.lower()
    # Turkish character normalization (covers both pre- and post-lowercase forms)
    tr_map = str.maketrans('şıığüöçŞİĞÜÖÇ', 'siiguocsiGuoc')
    s = s.translate(tr_map)
    # Collapse whitespace / punctuation leftovers
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


class RandomForestNeighborModel:
    """
    Random Forest classifier extended with neighbor assembly context features:
      - neighbor_assembly_m2    (total assembly area within 500 m buffer)
      - neighbor_assembly_count (number of assembly areas within 500 m buffer)

    Uses the same 70/30 train/test split produced by Data_Splitter.py.
    """

    OUTPUT_DIR = "ML_RF_Neighbor_Charts"
    NEIGHBOR_CSV = "izmir_mahalle_neighbor_assembly.csv"
    TRAIN_CSV = "output/Train_Dataset.csv"
    TEST_CSV = "output/Test_Dataset.csv"

    BASE_FEATURES = ['ILCE_encoded', 'NUFUS', 'LST_C', 'NDVI', 'NDBI']
    NEIGHBOR_FEATURES = ['neighbor_assembly_m2', 'neighbor_assembly_count']
    TARGET = 'AREA_CATEGORY'

    FEATURE_LABELS = {
        'ILCE_encoded': 'ILCE',
        'NUFUS': 'NUFUS',
        'LST_C': 'LST_C',
        'NDVI': 'NDVI',
        'NDBI': 'NDBI',
        'neighbor_assembly_m2': 'Neighbor Assembly m²',
        'neighbor_assembly_count': 'Neighbor Assembly Count',
    }

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.le_ilce = LabelEncoder()
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.all_features = self.BASE_FEATURES + self.NEIGHBOR_FEATURES

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_neighbor_lookup(neighbor_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize both mahalle name and ilce to build a composite key lookup."""
        nb = neighbor_df[['name', 'ilce_adi', 'neighbor_assembly_m2', 'neighbor_assembly_count']].copy()
        nb['_mkey'] = nb['name'].apply(_normalize_mahalle)
        nb['_mkey_ns'] = nb['_mkey'].str.replace(' ', '', regex=False)
        nb['_ikey'] = nb['ilce_adi'].apply(_normalize_mahalle)
        # Composite key: ilce + mahalle (most specific)
        nb['_composite'] = nb['_ikey'] + '|' + nb['_mkey']
        nb['_composite_ns'] = nb['_ikey'] + '|' + nb['_mkey_ns']
        return nb

    def _merge_neighbor_data(self, df: pd.DataFrame, neighbor_lookup: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['_mkey'] = df['MAHALLE'].apply(_normalize_mahalle)
        df['_mkey_ns'] = df['_mkey'].str.replace(' ', '', regex=False)
        df['_ikey'] = df['ILCE'].apply(_normalize_mahalle)
        df['_composite'] = df['_ikey'] + '|' + df['_mkey']
        df['_composite_ns'] = df['_ikey'] + '|' + df['_mkey_ns']

        nb_cols = ['neighbor_assembly_m2', 'neighbor_assembly_count']
        df['neighbor_assembly_m2'] = np.nan
        df['neighbor_assembly_count'] = np.nan

        def _make_map(nb, key_col):
            deduped = nb.groupby(key_col, as_index=False)[nb_cols].sum()
            return deduped.set_index(key_col)

        # Pass 1: exact composite key (ilce + mahalle name)
        p1_map = _make_map(neighbor_lookup, '_composite')
        hits = df['_composite'].map(p1_map['neighbor_assembly_m2'])
        matched_p1 = hits.notna()
        df.loc[matched_p1, 'neighbor_assembly_m2'] = hits[matched_p1].values
        df.loc[matched_p1, 'neighbor_assembly_count'] = df.loc[matched_p1, '_composite'].map(
            p1_map['neighbor_assembly_count']).values

        # Pass 2: no-space composite key (handles "YENIKOY" vs "Yeni Köy")
        unmatched = df['neighbor_assembly_m2'].isna()
        if unmatched.any():
            p2_map = _make_map(neighbor_lookup, '_composite_ns')
            hits2 = df.loc[unmatched, '_composite_ns'].map(p2_map['neighbor_assembly_m2'])
            matched_p2 = hits2.notna()
            df.loc[matched_p2[matched_p2].index, 'neighbor_assembly_m2'] = hits2[matched_p2].values
            df.loc[matched_p2[matched_p2].index, 'neighbor_assembly_count'] = (
                df.loc[matched_p2[matched_p2].index, '_composite_ns']
                .map(p2_map['neighbor_assembly_count']).values)

        # Pass 3: mahalle name only (no-space) — fallback without ilce
        unmatched = df['neighbor_assembly_m2'].isna()
        if unmatched.any():
            p3_map = _make_map(neighbor_lookup, '_mkey_ns')
            hits3 = df.loc[unmatched, '_mkey_ns'].map(p3_map['neighbor_assembly_m2'])
            matched_p3 = hits3.notna()
            df.loc[matched_p3[matched_p3].index, 'neighbor_assembly_m2'] = hits3[matched_p3].values
            df.loc[matched_p3[matched_p3].index, 'neighbor_assembly_count'] = (
                df.loc[matched_p3[matched_p3].index, '_mkey_ns']
                .map(p3_map['neighbor_assembly_count']).values)

        df = df.drop(columns=['_mkey', '_mkey_ns', '_ikey', '_composite', '_composite_ns'])
        df['neighbor_assembly_m2'] = df['neighbor_assembly_m2'].fillna(0.0)
        df['neighbor_assembly_count'] = df['neighbor_assembly_count'].fillna(0.0)
        return df

    # ------------------------------------------------------------------
    def load_data(self):
        self.train_df = pd.read_csv(self.TRAIN_CSV)
        self.test_df = pd.read_csv(self.TEST_CSV)
        neighbor_df = pd.read_csv(self.NEIGHBOR_CSV)

        neighbor_lookup = self._prepare_neighbor_lookup(neighbor_df)

        # Track join hits before merge (NaN = not found in lookup)
        self.train_df = self._merge_neighbor_data(self.train_df, neighbor_lookup)
        self.test_df = self._merge_neighbor_data(self.test_df, neighbor_lookup)

        total = len(self.train_df) + len(self.test_df)
        # A row was "found" if neighbor_assembly_m2 came from the CSV (could be 0 legitimately)
        # We track this via a sentinel: unmatched rows are filled with NaN then 0 AFTER we report
        # Instead, re-derive: check the lookup keys directly
        nb_keys_composite = set(neighbor_lookup['_composite'].tolist())
        nb_keys_composite_ns = set(neighbor_lookup['_composite_ns'].tolist())
        nb_keys_ns = set(neighbor_lookup['_mkey_ns'].tolist())

        all_df = pd.concat([self.train_df, self.test_df])
        all_df['_mkey'] = all_df['MAHALLE'].apply(_normalize_mahalle)
        all_df['_mkey_ns'] = all_df['_mkey'].str.replace(' ', '', regex=False)
        all_df['_ikey'] = all_df['ILCE'].apply(_normalize_mahalle)
        all_df['_composite'] = all_df['_ikey'] + '|' + all_df['_mkey']
        all_df['_composite_ns'] = all_df['_ikey'] + '|' + all_df['_mkey_ns']

        found = (
            all_df['_composite'].isin(nb_keys_composite)
            | all_df['_composite_ns'].isin(nb_keys_composite_ns)
            | all_df['_mkey_ns'].isin(nb_keys_ns)
        )
        print(f"Neighbor join: {found.sum()}/{total} rows matched ({found.mean():.1%})")

        all_ilce = pd.concat([self.train_df['ILCE'], self.test_df['ILCE']])
        self.le_ilce.fit(all_ilce)
        self.train_df['ILCE_encoded'] = self.le_ilce.transform(self.train_df['ILCE'])
        self.test_df['ILCE_encoded'] = self.le_ilce.transform(self.test_df['ILCE'])

        self.X_train = self.train_df[self.all_features]
        self.y_train = self.train_df[self.TARGET]
        self.X_test = self.test_df[self.all_features]
        self.y_test = self.test_df[self.TARGET]

        print(f"Total neighborhoods: {len(self.train_df) + len(self.test_df)}")
        print(f"Features: {', '.join(self.all_features)}")
        print(f"Train: {len(self.X_train)} | Test: {len(self.X_test)}")

    # ------------------------------------------------------------------
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete.")

    # ------------------------------------------------------------------
    def evaluate(self):
        self.y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)

        print("\n             MODEL PERFORMANCE REPORT (+ Neighbor Features)             ")
        print("=========================================================================")
        print(f"Overall Accuracy : {accuracy:.2%}")
        print("=========================================================================\n")
        print(report)
        print("=========================================================================\n")
        return accuracy, report

    # ------------------------------------------------------------------
    def _plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.model.classes_)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.model.classes_,
                    yticklabels=self.model.classes_)
        plt.title('RF + Neighbor Context: Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
        plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Category', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        path = f"{self.OUTPUT_DIR}/1_RF_Neighbor_Confusion_Matrix.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Confusion Matrix saved → {path}")

    # ------------------------------------------------------------------
    def plot_feature_importance(self):
        importances = self.model.feature_importances_ * 100
        labels = [self.FEATURE_LABELS[f] for f in self.all_features]
        importance_df = pd.DataFrame({
            'Feature': labels,
            'Importance (%)': importances
        }).sort_values('Importance (%)', ascending=False)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x='Importance (%)',
            y='Feature',
            data=importance_df,
            hue='Feature',
            legend=False,
            palette='viridis'
        )
        for p in ax.patches:
            width = p.get_width()
            plt.text(width + 0.5, p.get_y() + p.get_height() / 2.0 + 0.05,
                     f'{width:.1f}%', ha='left', va='center', fontsize=10)

        plt.xlim(0, max(importances) * 1.2)
        plt.title('RF + Neighbor Context: Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Impact Weight on Model Decisions (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = f"{self.OUTPUT_DIR}/2_RF_Neighbor_Feature_Importance.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Feature Importance saved → {path}")

    # ------------------------------------------------------------------
    def plot_performance_summary(self):
        from sklearn.metrics import classification_report
        report_dict = classification_report(
            self.y_test, self.y_pred, output_dict=True
        )
        accuracy = accuracy_score(self.y_test, self.y_pred)

        classes = [k for k in report_dict if k not in ('accuracy', 'macro avg', 'weighted avg')]
        rows = []
        for cls in classes:
            m = report_dict[cls]
            rows.append([
                cls.replace('_', ' '),
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1-score']:.3f}",
                int(m['support'])
            ])

        for avg_key in ('macro avg', 'weighted avg'):
            m = report_dict[avg_key]
            rows.append([
                avg_key.title(),
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1-score']:.3f}",
                int(m['support'])
            ])

        col_labels = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']

        fig = plt.figure(figsize=(13, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], figure=fig)

        # --- Left: table ---
        ax_table = fig.add_subplot(gs[0])
        ax_table.axis('off')
        tbl = ax_table.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)

        header_color = '#2c7bb6'
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor(header_color)
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')

        neighbor_row_indices = list(range(1, len(classes) + 1))
        for i, row_idx in enumerate(neighbor_row_indices):
            bg = '#f0f7ff' if i % 2 == 0 else '#ffffff'
            for j in range(len(col_labels)):
                tbl[(row_idx, j)].set_facecolor(bg)

        for i, avg_row in enumerate(range(len(classes) + 1, len(classes) + 3)):
            bg = '#d4edda' if i == 0 else '#c8e6c9'
            for j in range(len(col_labels)):
                tbl[(avg_row, j)].set_facecolor(bg)
                tbl[(avg_row, j)].set_text_props(fontweight='bold')

        ax_table.set_title(
            'RF + Neighbor Context: Per-Class Performance',
            fontsize=12, fontweight='bold', pad=10
        )

        # --- Right: accuracy gauge ---
        ax_gauge = fig.add_subplot(gs[1])
        ax_gauge.axis('off')
        ax_gauge.text(0.5, 0.65, f"{accuracy:.1%}",
                      ha='center', va='center', fontsize=44, fontweight='bold',
                      color='#2c7bb6', transform=ax_gauge.transAxes)
        ax_gauge.text(0.5, 0.45, 'Overall Accuracy',
                      ha='center', va='center', fontsize=13, fontweight='bold',
                      color='#333333', transform=ax_gauge.transAxes)
        ax_gauge.text(0.5, 0.33, f"Train: {len(self.X_train)}  |  Test: {len(self.X_test)}",
                      ha='center', va='center', fontsize=10, color='#555555',
                      transform=ax_gauge.transAxes)
        ax_gauge.text(0.5, 0.22, f"Features: {len(self.all_features)}",
                      ha='center', va='center', fontsize=10, color='#555555',
                      transform=ax_gauge.transAxes)
        ax_gauge.set_title('Summary', fontsize=12, fontweight='bold', pad=10)

        plt.tight_layout()
        path = f"{self.OUTPUT_DIR}/3_RF_Neighbor_Performance_Summary.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance Summary saved → {path}")

    # ------------------------------------------------------------------
    def run(self):
        self.load_data()
        self.train()
        self.evaluate()
        self._plot_confusion_matrix()
        self.plot_feature_importance()
        self.plot_performance_summary()
        print(f"\nAll outputs saved to: {self.OUTPUT_DIR}/")


if __name__ == "__main__":
    model = RandomForestNeighborModel()
    model.run()
