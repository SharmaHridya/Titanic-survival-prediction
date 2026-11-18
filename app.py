import numpy as np
import pandas as pd
import streamlit as st


model = np.load("model.npz", allow_pickle=True)
w = model["w"]
b = float(model["b"])
mu = model["mu"]
sigma = model["sigma"]
columns = model["columns"].tolist()
AGE_BY_TITLE = pd.Series(model["age_by_title"].item())
FARE_MEDIAN = float(model["fare_median"])


def extract_title(names):
    title = names.str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common = ["Mr", "Mrs", "Miss", "Master"]
    return title.where(title.isin(common), "Rare")


def prepare(df):
    out = pd.DataFrame(index=df.index)
    out["Sex"] = (df["Sex"] == "female").astype(int)

    title = extract_title(df["Name"])
    out["Age"] = df["Age"].fillna(title.map(AGE_BY_TITLE))
    out["LogFare"] = np.log1p(df["Fare"].fillna(FARE_MEDIAN))
    out["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["HasCabin"] = df["Cabin"].notna().astype(int)

    embarked = df["Embarked"].fillna("S")

    for c in [2, 3]:
        out[f"Pclass_{c}"] = (df["Pclass"] == c).astype(int)
    for t in ["Mrs", "Miss", "Master", "Rare"]:
        out[f"Title_{t}"] = (title == t).astype(int)
    for e in ["Q", "S"]:
        out[f"Embarked_{e}"] = (embarked == e).astype(int)

    return out


def add_extras(X_df):
    X = X_df.copy()
    X["Sex_x_Pclass3"] = X["Sex"] * X["Pclass_3"]
    X["Age_sq"] = X["Age"] ** 2
    X["Age_x_Sex"] = X["Age"] * X["Sex"]
    X["Fare_x_Sex"] = X["LogFare"] * X["Sex"]
    X["FamilySize_sq"] = X["FamilySize"] ** 2
    return X


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


st.title("Would I have survived the Titanic disaster?")
st.write("Enter passenger details to see the model’s estimated survival probability.")

pclass = st.selectbox("Passenger class", [1, 2, 3])
sex = st.selectbox("Sex", ["female", "male"])
age = st.slider("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, value=32.0, step=1.0)
family_size = st.slider("Family size (including you)", 1, 11, 1)
port = st.selectbox("Port of embarkation", ["S", "C", "Q"])

if st.button("Estimate"):
    title = "Miss" if sex == "female" else "Mr"
    name = f"Passenger, {title}. Example"

    row = pd.DataFrame([{
        "Name": name,
        "Sex": sex,
        "Age": age,
        "Fare": fare,
        "SibSp": family_size - 1,
        "Parch": 0,
        "Cabin": np.nan,
        "Pclass": pclass,
        "Embarked": port,
    }])

    features = add_extras(prepare(row)).reindex(
        columns=columns,
        fill_value=0
    )
    x = features.to_numpy(dtype=float)
    x_scaled = (x - mu) / sigma

    probability = sigmoid(x_scaled @ w + b).item()
    st.metric("Estimated survival probability", f"{probability:.1%}")
    st.caption("This is a model estimate based on patterns in the Titanic training data.")