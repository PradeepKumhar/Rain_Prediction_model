import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # Handle missing values (replace with median)
    for col in ['Temperature_C', 'Humidity_%', 'Wind_Speed_kmh', 'Pressure_hPa', 'Cloud_Cover_%', 'Precipitation_mm']:
        data[col] = data[col].fillna(data[col].median())

    # Encode categorical variable
    data['Rain_Today'] = data['Rain_Today'].map({'Yes': 1, 'No': 0})

    # Remove duplicates
    data = data.drop_duplicates().reset_index(drop=True)

    # Feature Scaling
    scaler = StandardScaler()
    features = ['Temperature_C', 'Humidity_%', 'Wind_Speed_kmh', 'Pressure_hPa', 'Cloud_Cover_%', 'Precipitation_mm']
    data[features] = scaler.fit_transform(data[features])

    return data, scaler  # Returning scaler for future use

if __name__ == "__main__":
    data, scaler = load_and_clean_data("dataset.csv")
    print("Data Processing Complete ✅")
