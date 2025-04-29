import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu, gaussian
from skimage import measure
import cv2
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import warnings
import re
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    img_blur = gaussian(img, sigma=1.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply((img_blur * 255 / img_blur.max()).astype(np.uint8))
    preprocessed_image = resize(img_enhanced, (512, 512), preserve_range=True, anti_aliasing=True)
    preprocessed_image = (preprocessed_image - preprocessed_image.min()) / (preprocessed_image.max() - preprocessed_image.min())
    preprocessed_image_rgb = np.stack([preprocessed_image] * 3, axis=-1)
    return preprocessed_image_rgb

def extract_contour_features(preprocessed_image):
    features = {}
    gray_image = preprocessed_image[:, :, 0]
    edges = cv2.Canny((gray_image * 255).astype(np.uint8), 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        features['mean_radius'] = np.mean(np.sqrt((gray_image.shape[0]/2)**2 + (gray_image.shape[1]/2)**2))
        features['mean_perimeter'] = 0
        features['mean_texture'] = np.std(gray_image)
        features['smoothness'] = 0
        features['compactness'] = 0
        features['symmetry'] = 0
    else:
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            features['mean_radius'] = 0
            features['mean_perimeter'] = 0
            features['mean_texture'] = np.std(gray_image)
            features['smoothness'] = 0
            features['compactness'] = 0
            features['symmetry'] = 0
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distances = [np.sqrt((p[0][0] - cx)**2 + (p[0][1] - cy)**2) for p in contour]
            features['mean_radius'] = np.mean(distances)
            features['mean_perimeter'] = cv2.arcLength(contour, True)
            mask = np.zeros_like(gray_image, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            features['mean_texture'] = np.std(gray_image[mask == 255])
            perimeter_points = contour.squeeze()
            if len(perimeter_points) > 1:
                diff = np.diff(perimeter_points, axis=0)
                features['smoothness'] = 1 / (np.std(diff) + 1e-6)
            else:
                features['smoothness'] = 0
            area = cv2.contourArea(contour)
            features['compactness'] = (features['mean_perimeter']**2) / (area + 1e-6)
            moments = cv2.HuMoments(M).flatten()
            features['symmetry'] = moments[1]
    return features

def extract_traditional_features(preprocessed_image):
    features = {}
    gray_image = preprocessed_image[:, :, 0]
    glcm = graycomatrix((gray_image * 255).astype(np.uint8), distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
    features['glcm_correlation'] = np.mean(graycoprops(glcm, 'correlation'))
    features['glcm_energy'] = np.mean(graycoprops(glcm, 'energy'))
    features['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
    features['glcm_dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
    features['intensity_mean'] = np.mean(gray_image)
    features['intensity_std'] = np.std(gray_image)
    features['intensity_median'] = np.median(gray_image)
    features['intensity_skewness'] = skew(gray_image.flatten())
    features['intensity_kurtosis'] = kurtosis(gray_image.flatten())
    features['intensity_p10'] = np.percentile(gray_image, 10)
    features['intensity_p90'] = np.percentile(gray_image, 90)
    features['entropy'] = shannon_entropy(gray_image)
    edges = cv2.Canny((gray_image * 255).astype(np.uint8), 100, 200)
    features['edge_density'] = np.sum(edges) / (gray_image.shape[0] * gray_image.shape[1])
    return features

def extract_deep_features(preprocessed_image):
    patch = resize(preprocessed_image, (224, 224), preserve_range=True, anti_aliasing=True)
    patch = (patch * 255).astype(np.uint8)
    if patch.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape: {patch.shape}")
    img = image.img_to_array(patch)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = base_model.predict(img, verbose=0)
    return features.flatten()

def extract_engineered_features(traditional_features):
    engineered_features = {}
    engineered_features['intensity_mean_squared'] = traditional_features['intensity_mean'] ** 2
    engineered_features['glcm_contrast_energy'] = traditional_features['glcm_contrast'] * traditional_features['glcm_energy']
    engineered_features['intensity_std_skewness'] = traditional_features['intensity_std'] * traditional_features['intensity_skewness']
    engineered_features['edge_density_entropy'] = traditional_features['edge_density'] * traditional_features['entropy']
    return engineered_features

def map_diagnoses(features_df, diagnosis_file='CBIS-DDSM/calc_case_description_train_set.csv'):
    """
    Map diagnoses to extracted features using normalized paths.
    """
    diagnosis_df = pd.read_csv(diagnosis_file)
    
    # Normalize paths from diagnosis_df (extract series ID)
    def normalize_calc_path(path):
        segments = path.split('/')
        if len(segments) >= 3:
            series_id = segments[-2]
            if re.match(r'1\.3\.6\.1\.4\.1\.\d+', series_id):
                return f'e:/codes/paper codes/CBIS-DDSM/jpeg\\{series_id}'
        return None
    diagnosis_df['normalized_path'] = diagnosis_df['image file path'].apply(normalize_calc_path)
    
    # Normalize paths from features_df (clean image_path)
    def clean_features_path(path):
        pattern = r'\\[^\\]+\.jpg$'
        return re.sub(pattern, '', path)
    features_df['normalized_path'] = features_df['image_path'].apply(clean_features_path)
    
    # Merge on normalized_path
    features_df = features_df.merge(diagnosis_df[['normalized_path', 'pathology']], on='normalized_path', how='left')
    missing_diagnoses = features_df['pathology'].isna().sum()
    if missing_diagnoses > 0:
        print(f"Warning: {missing_diagnoses} images could not be matched to diagnoses. Check path alignment.")
        print("Sample normalized_paths in features_df:", features_df['normalized_path'].head().tolist())
        print("Sample normalized_paths in diagnosis_df:", diagnosis_df['normalized_path'].head().tolist())
    return features_df

def save_extracted_features(image_dir, output_file, diagnosis_file='CBIS-DDSM/calc_case_description_train_set.csv'):
    all_features = []
    all_deep_features = []
    
    print("Scanning JPEG subfolders for feature extraction...")
    for root, dirs, files in os.walk(image_dir):
        for file in sorted(files):
            if file.endswith('.jpg'):
                full_image_path = os.path.join(root, file)
                print(f"Processing image: {full_image_path}")
                image_id = os.path.basename(file).replace('.jpg', '')
                
                preprocessed_image = preprocess_image(full_image_path)
                if preprocessed_image is None:
                    continue
                
                contour_features = extract_contour_features(preprocessed_image)
                if contour_features is None:
                    continue
                
                traditional_features = extract_traditional_features(preprocessed_image)
                traditional_features.update(contour_features)
                
                deep_features = extract_deep_features(preprocessed_image)
                all_deep_features.append(deep_features)
                
                for i, val in enumerate(deep_features[:10]):
                    traditional_features[f'deep_feature_{i+1}'] = val
                
                engineered_features = extract_engineered_features(traditional_features)
                traditional_features.update(engineered_features)
                
                traditional_features['image_path'] = full_image_path
                traditional_features['image_id'] = image_id
                
                all_features.append(traditional_features)
    
    if not all_features:
        print("No features extracted. Check JPEG files and paths.")
        return
    
    features_df = pd.DataFrame(all_features)
    
    if all_deep_features:
        deep_features_array = np.array(all_deep_features)
        scaler = RobustScaler()
        deep_features_scaled = scaler.fit_transform(deep_features_array)
        pca = PCA(n_components=10, random_state=42)
        deep_features_pca = pca.fit_transform(deep_features_scaled)
        for i in range(10):
            features_df[f'deep_feature_{i+1}'] = deep_features_pca[:, i]
    
    features_df = map_diagnoses(features_df, diagnosis_file)
    
    numeric_features = features_df.select_dtypes(include=[np.number]).columns.drop(['pathology'], errors='ignore')
    correlation_matrix = features_df[numeric_features].corr(method='pearson')
    high_correlations = np.where(np.abs(correlation_matrix) > 0.9)
    high_correlations = [(correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
                        for i, j in zip(*high_correlations) if i != j]
    if high_correlations:
        print("High correlations (>0.9) detected:")
        for feature1, feature2, corr in high_correlations:
            print(f"{feature1} vs {feature2}: {corr:.3f}")
        features_to_drop = set()
        for feature1, feature2, _ in high_correlations:
            features_to_drop.add(feature2)
        features_df = features_df.drop(columns=list(features_to_drop.intersection(numeric_features)))
    
    numeric_features = features_df.select_dtypes(include=[np.number]).columns.drop(['pathology'], errors='ignore')
    scaler = RobustScaler()
    features_df[numeric_features] = scaler.fit_transform(features_df[numeric_features])
    
    features_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Extracted features saved to {output_file}")
    print(f"Number of features extracted: {len(numeric_features)}")

def main():
    image_dir = "e:/codes/paper codes/CBIS-DDSM/jpeg"
    output_file = 'extracted_cbis_features_updated.csv'
    diagnosis_file = 'CBIS-DDSM/calc_case_description_train_set.csv'
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory '{image_dir}' not found.")
        return
    
    save_extracted_features(image_dir, output_file, diagnosis_file)

if __name__ == "__main__":
    main()