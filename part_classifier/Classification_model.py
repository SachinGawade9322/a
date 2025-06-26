import pandas as pd
import numpy as np
import re
import json
from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

df = pd.read_csv(r"E:\datta_enterprises\AI_Models\Processed_data\Train_data\Final_all_images_structured_output_json_headers.csv")

df["description"] = df["description"].str.lower().str.strip()
df["description"] = df["description"].replace({
    "5mm section to be threaded": "thread",
    "threaded hole": "thread",
    "surface finish of indicated faces": "surface finish",
})

def extract_dimensions(dim_text):
    features = {
        "Diameter": np.nan, "Depth": np.nan, "Count": np.nan,
        "Radius": np.nan, "Width": np.nan, "Height": np.nan,
        "Length": np.nan, "Size": np.nan, "Dimension": np.nan
    }
    if pd.isna(dim_text):
        return features
    pairs = dim_text.split(";")
    for p in pairs:
        if ":" in p:
            k, v = p.strip().split(":")
            v_clean = re.findall(r"[\d.]+", v.replace("\u00d8", "").strip())
            if v_clean:
                try:
                    features[k.strip()] = float(v_clean[0])
                except:
                    continue
    return features

dim_df = pd.DataFrame(df["dimensions"].apply(extract_dimensions).to_list())
df = pd.concat([df, dim_df], axis=1)

from transformers import logging
logging.set_verbosity_error()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

descriptions = df["description"].astype(str).tolist()
tokens = tokenizer(descriptions, padding=True, truncation=True, max_length=32, return_tensors="tf")
bert_outputs = bert_model(tokens)
cls_embeddings = bert_outputs.last_hidden_state[:, 0, :].numpy()

structured_cols = ["type", "material", "Diameter", "Depth", "Count",
                   "Radius", "Width", "Height", "Length", "Size", "Dimension"]

type_encoder = LabelEncoder()
df["type"] = type_encoder.fit_transform(df["type"].astype(str))
material_encoder = LabelEncoder()
df["material"] = material_encoder.fit_transform(df["material"].astype(str))

scaler = StandardScaler()
df.fillna(0, inplace=True)
structured_X = scaler.fit_transform(df[structured_cols])

# ------------------------------
# 5. Combine Features
# ------------------------------
X_combined = np.concatenate([cls_embeddings, structured_X], axis=1)

# ------------------------------
# 6. Label Encoding and Weights
# ------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["description"].astype(str))
y_cat = to_categorical(y)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

# ------------------------------
# 7. Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_cat, test_size=0.2, random_state=42)

# ------------------------------
# 8. Build & Train Model
# ------------------------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_combined.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, class_weight=class_weight_dict)

# ------------------------------
# 10. Predict on JSON-based Inputs
# ------------------------------
json_path = r"E:\datta_enterprises\backend\detect\feature_extraction.json"
with open(json_path) as f:
    json_data = json.load(f)

material = json_data["gemini_analysis"]["meta_data"]["material"]
features = json_data["gemini_analysis"]["features"]

model_inputs = []
for f in features:
    desc = f.get("description", f["type"])
    dims = {
        "Diameter": float(re.findall(r"[\d.]+", f.get("diameter", "0"))[0]) if "diameter" in f else 0.0,
        "Depth": float(f.get("depth", 0.0)),
        "Count": float(f.get("count", 1.0)),
        "Radius": float(f.get("radius", 0.0)),
        "Width": float(f.get("width", 0.0)),
        "Height": float(f.get("height", 0.0)),
        "Length": float(re.findall(r"[\d.]+", f.get("length", "0"))[0]) if "length" in f else 0.0,
        "Size": float(0.0),
        "Dimension": float(f.get("dimension", 0.0))
    }
    input_row = {
        "description": desc,
        "type": f["type"],
        "material": material,
        **dims
    }
    model_inputs.append(input_row)

for item in model_inputs:
    item["description"] = item["description"].lower().strip().replace(
        "5mm section to be threaded", "thread"
    ).replace("threaded hole", "thread").replace(
        "surface finish of indicated faces", "surface finish"
    )

descriptions = [item["description"] for item in model_inputs]
tokens = tokenizer(descriptions, padding=True, truncation=True, max_length=32, return_tensors="tf")
bert_embeddings = bert_model(tokens).last_hidden_state[:, 0, :].numpy()

test_df = pd.DataFrame(model_inputs)
test_df["type"] = type_encoder.transform(test_df["type"].astype(str))
test_df["material"] = material_encoder.transform(test_df["material"].astype(str))
test_df.fillna(0, inplace=True)
structured_features = scaler.transform(test_df[structured_cols])

X_final = np.concatenate([bert_embeddings, structured_features], axis=1)
predictions = model.predict(X_final)
pred_indices = np.argmax(predictions, axis=1)
pred_labels = label_encoder.inverse_transform(pred_indices)

for i, item in enumerate(model_inputs):
    print(f"\n📦 Feature {i+1}:")
    print(f"📝 Description: {item['description']}")
    print(f"🔧 Type: {item['type']}")
    print(f"✅ Predicted Subpart Label: {pred_labels[i]}")
