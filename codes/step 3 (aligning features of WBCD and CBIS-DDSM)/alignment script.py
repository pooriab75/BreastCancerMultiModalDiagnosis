import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Load datasets (replace paths with your actual file locations)
wbcd_df = pd.read_csv('step 1/wisconsin/wisconsin_breast_cancer_dataset.csv')  # WBCD file
cbis_df = pd.read_csv('step 2/combined_dataset_clean_ddsm_train_and_test.csv')       # CBIS-DDSM file

# Step 1: Process WBCD
wbcd_df = wbcd_df.drop(columns=['id'])  # Drop identifier
wbcd_df['diagnosis'] = wbcd_df['diagnosis'].map({'B': 0, 'M': 1})  # Map to binary
wbcd_features = wbcd_df.drop(columns=['diagnosis'])  # Features only
wbcd_labels = wbcd_df['diagnosis']

# Step 2: Process CBIS-DDSM
# Keep all rows (no deduplication, as duplicates are different views)
# Drop identifiers
cbis_df = cbis_df.drop(columns=['image_path', 'image_id'])

# Map pathology to binary (BENIGN_WITHOUT_CALLBACK -> BENIGN)
cbis_df['pathology'] = cbis_df['pathology'].map({
    'BENIGN': 0,
    'BENIGN_WITHOUT_CALLBACK': 0,
    'MALIGNANT': 1
})

# Extract labels
cbis_labels = cbis_df['pathology']
cbis_features = cbis_df.drop(columns=['pathology'])

# Step 3: Align CBIS-DDSM to WBCD feature set
wbcd_feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
    'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# Rename CBIS-DDSM features to match WBCD where possible
cbis_features = cbis_features.rename(columns={
    'mean_radius': 'radius_mean',
    'mean_perimeter': 'perimeter_mean',
    'mean_texture': 'texture_mean',
    'smoothness': 'smoothness_mean',      # Direct match
    'compactness': 'compactness_mean',    # Direct match
    'symmetry': 'symmetry_mean'           # Direct match
})

# Create a new DataFrame with WBCD feature names
cbis_aligned = pd.DataFrame(columns=wbcd_feature_names)

# Copy over matching features
for col in cbis_features.columns:
    if col in wbcd_feature_names:
        cbis_aligned[col] = cbis_features[col]

# Fill missing features with NaN (to be imputed later if we want to)
cbis_aligned = cbis_aligned.fillna(np.nan)

# Step 4: Combine datasets
combined_features = pd.concat([wbcd_features, cbis_aligned], ignore_index=True)
combined_labels = pd.concat([wbcd_labels, cbis_labels], ignore_index=True)
# Normalize data
scaler = StandardScaler()
combined_features_scaled = scaler.fit_transform(combined_features)

# KNN imputation
imputer = KNNImputer(n_neighbors=5, weights='distance')
combined_features_imputed = imputer.fit_transform(combined_features_scaled)

# Reverse scaling
combined_features = pd.DataFrame(scaler.inverse_transform(combined_features_imputed), 
                                 columns=wbcd_feature_names)


# Save aligned dataset
combined_df = pd.concat([combined_labels.rename('diagnosis'), combined_features], axis=1)
combined_df.to_csv('aligned_breast_cancer_dataset.csv', index=False)

print("Aligned dataset saved as 'aligned_breast_cancer_dataset.csv'")
print(f"Shape: {combined_df.shape}")
print(f"Features: {list(combined_df.columns)}")