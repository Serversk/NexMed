import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize


train="train"
test="test"
IMG_SIZE = (64,64)

#labels used in the dataset named: Abdelrauofkhanfar/Skin-diseases
LABELS={
    0:"Actinic Keratosis Basal Cell Carcinoma",
    1:"Nail Fungus",
    2:"Psoriasis Lichen Planus",
    3:"Seborrheic Keratoses",
    4:"Tinea Ringworm Candidiasis",
    5:"Warts Molluscum",
    6:"Acne and Rosacea"
}
#list of datasets used
datasets=[
    "WafaaFraih/rocov2-mri-20250831-190356"
]

def load_data(HF_DATASET="WafaaFraih/rocov2-mri-20250831-190356"):
    dataset = load_dataset(HF_DATASET)
    x_train,y_train,x_test,y_test=[],[],[],[]

    for example in tqdm(dataset[train], desc="Processing images", total=len(dataset[train])):
        img = np.array(example["image"]) /255
        label=example["caption"]

        img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
        x_train.append(img_resized)
        y_train.append(label)

    if "test" in dataset:
        for example in tqdm(dataset[test], desc="Processing images", total=len(dataset[test])):
            img = np.array(example["image"]) /255
            label=example["caption"]
            img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
            x_test.append(img_resized)
            y_test.append(label)
    else:
        #calculate the last n% of the data then perform split on the data
        n=10
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=n/100, random_state=42, stratify=y_train)        

    return ((x_train,y_train),(x_test,y_test))

def save_data(x_train, y_train, x_test, y_test, train_path="Backend/train.csv",test_path="Backend/test.csv"):
    # Flatten images for CSV storage
    x_train_flat = [img.flatten() for img in tqdm(x_train, desc="Flattening training images",total=len(x_train))]
    x_test_flat = [img.flatten() for img in tqdm(x_test, desc="Flattening testing images",total=len(x_test))]

    # Create DataFrames
    train_df = pd.DataFrame(x_train_flat)
    train_df['label'] = y_train

    test_df = pd.DataFrame(x_test_flat)
    test_df['label'] = y_test

    # Save to CSV
    train_df.to_csv(train_path, index=False,encoding="utf-8")
    test_df.to_csv(test_path, index=False,encoding="utf-8")
    print(f"Data saved to {train_path} and {test_path}")

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    save_data(x_train, y_train, x_test, y_test)