import pandas as pd
import re

# Load the CSV files
calc_train_df = pd.read_csv('CBIS-DDSM/calc_case_description_train_set.csv')
calc_test_df = pd.read_csv('CBIS-DDSM/calc_case_description_test_set.csv')
features_df = pd.read_csv('step 2/extracted_cbis_features_updated.csv')


# Function to extract the series ID from calc paths
def normalize_path(path):
    """
    Extract the series ID from calc path and format it to match features_df
    Input: Path like 'Calc-Test_P_00038_LEFT_CC/1.3.6.1.4.1.../000000.dcm'
    Output: Path like 'e:/codes/paper codes/CBIS-DDSM/jpeg\1.3.6.1.4.1...'
    """
    segments = path.split('/')
    if len(segments) >= 3:
        series_id = segments[-2]  # Second-to-last segment is the series ID
        if re.match(r'1\.3\.6\.1\.4\.1\.\d+', series_id):
            return f'e:/codes/paper codes/CBIS-DDSM/jpeg\\{series_id}'
    return None

# Function to clean features_df path (remove image number and .jpg)
def clean_features_path(path):
    """
    Clean features_df path by removing image number and .jpg
    Input: Path like 'e:/codes/paper codes/CBIS-DDSM/jpeg\1.3.6.1.4.1...\1-263.jpg'
    Output: Path like 'e:/codes/paper codes/CBIS-DDSM/jpeg\1.3.6.1.4.1...'
    """
    pattern = r'\\[^\\]+\.jpg$'
    return re.sub(pattern, '', path)

# Combine calcification datasets (train + test)
calc_combined_df = pd.concat([calc_train_df, calc_test_df], ignore_index=True)
calc_combined_df['normalized_path'] = calc_combined_df['image file path'].apply(normalize_path)

# Normalize paths in features_df
features_df['normalized_path'] = features_df['image_path'].apply(clean_features_path)

# Merge features_df with calc data
merged_df = pd.merge(
    features_df,
    calc_combined_df[['normalized_path', 'pathology']],
    on='normalized_path',
    how='left',
    suffixes=('_features', '_calc')
)

# Combine pathology columns: use calc pathology where available, otherwise features pathology
merged_df['pathology'] = merged_df['pathology_calc'].fillna(merged_df['pathology_features'])

# Drop rows with NaN pathology
final_dataset = merged_df.dropna(subset=['pathology'])

# Drop temporary columns
final_dataset = final_dataset.drop(columns=['pathology_features', 'pathology_calc', 'normalized_path'])

# Display some information about the result
print(f"Number of records in final dataset: {len(final_dataset)}")
print(f"Columns in final dataset: {list(final_dataset.columns)}")
print("\nFirst few rows of the final dataset:")
print(final_dataset.head())

# Save the result to a new CSV file
final_dataset.to_csv('combined_dataset_clean_train_and_test.csv', index=False)
print("\nDataset saved to 'combined_dataset_clean.csv'")

# Show pathology distribution
print("\nPathology distribution after dropping NaN:")
print(final_dataset['pathology'].value_counts(dropna=False))

# Debug information
print("\nSample normalized paths from calc_combined_df:")
print(calc_combined_df['normalized_path'].head())
print("\nSample original paths from calc_combined_df:")
print(calc_combined_df['image file path'].head())
print("\nSample normalized paths from features_df:")
print(features_df['normalized_path'].head())
print("\nSample original paths from features_df:")
print(features_df['image_path'].head())