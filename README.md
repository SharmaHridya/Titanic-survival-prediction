# Titanic-survival-prediction
# Titanic Survival Prediction

A small machine-learning project that trains a logistic regression model on Titanic passenger data and provides an interactive Streamlit app for estimating survival probability from passenger details.

## How it works

The notebook prepares passenger features such as sex, age, fare, family size, passenger class, title, and embarkation port. It adds a few interaction and squared features, standardizes them using statistics learned from the training split, and trains logistic regression with gradient descent.

The Streamlit app loads the saved model parameters and applies the same feature preparation and scaling steps to one passenger at a time. It displays the model's estimated probability of survival.

## Requirements

- Python 3
- NumPy
- pandas
- scikit-learn
- Streamlit

Install the packages with:

```bash
python3 -m pip install numpy pandas scikit-learn streamlit
```

## Save the trained model

After training the final model in your notebook, save the model parameters and preprocessing values. This example assumes these variables already exist: `w`, `b`, `mu`, `sigma`, `X_all_extra`, `AGE_BY_TITLE`, and `FARE_MEDIAN`.

```python
import numpy as np

np.savez(
    "model.npz",
    w=w,
    b=b,
    mu=mu,
    sigma=sigma,
    columns=X_all_extra.columns.to_numpy(),
    age_by_title=AGE_BY_TITLE.to_dict(),
    fare_median=FARE_MEDIAN,
)
```

Keep `model.npz` in the same folder as `app.py`.

## Run the app

From that folder, run:

```bash
python3 -m streamlit run app.py
```

Streamlit will print a local URL to open in your browser. Enter passenger details and select **Estimate** to see the model's predicted survival probability.

## Project files

- `app.py` — Streamlit application.
- `model.npz` — saved model weights and preprocessing data; generated from the notebook.
- Notebook — data preparation, model training, validation, and model export.

## Notes

- The app must use the same feature creation and column order used during training. `model.npz` stores the training column names and scaling values for this purpose.
- The app uses a default title based on the selected sex (`Miss` for female, `Mr` for male), because title is not a separate input.
- The displayed probability is an estimate from the trained model, not a guarantee.
- Ensure the dataset and model files are used in accordance with their respective licenses and terms.
