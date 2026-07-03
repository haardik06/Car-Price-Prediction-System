from sklearn.preprocessing import LabelEncoder

def encode_features(data):

    # Create a copy
    df_encoded = data.copy()

    # Columns to encode
    categorical_columns = [
        "brand",
        "model",
        "seller_type",
        "fuel_type",
        "transmission_type"
    ]

    # Dictionary to store encoders
    label_encoders = {}

    # Encode each categorical column
    for column in categorical_columns:

        encoder = LabelEncoder()

        df_encoded[column] = encoder.fit_transform(df_encoded[column])

        label_encoders[column] = encoder

    return df_encoded, label_encoders