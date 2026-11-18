
🚢 Titanic Survival Predictor
Predicting whether a Titanic passenger survived, using logistic regression written from scratch in NumPy, then checked against scikit-learn.

Built as a first project after Course 1 of the Machine Learning Specialization (Andrew Ng, DeepLearning.AI / Stanford). Data from Kaggle's Titanic - Machine Learning from Disaster competition.

Table of contents
Results
Project structure
Setup and how to run
Workflow: the big picture
Workflow: stage by stage
How the model makes one prediction
How the model learns (training loop)
Tuning: choosing α and λ
Key takeaways
Next steps
Results
✍️ Fill this in after running the notebook.

Model	Features	λ	Train acc	Validation acc	Kaggle score
Baseline: "everyone died"	—	—	0.616	—	—
Baseline: "women survive, men die"	sex	—	0.787	—	~0.766
Logistic regression (from scratch)	14	0			
+ interaction/polynomial features	19	0			
+ regularization (best λ)	19				
scikit-learn LogisticRegression	19	same			—
Most important features (from the learned weights): fill in, e.g. Sex (+), Pclass_3 (−), Title_Master (+)…

Project structure
titanic/
├── data/
│   ├── train.csv          # 891 passengers, with the Survived label
│   └── test.csv           # 418 passengers, no label (for the Kaggle submission)
├── titanic.ipynb          # all the code: EDA → features → model → submission
├── submission.csv         # predictions uploaded to Kaggle
├── README.md              # this file
└── .venv/                 # Python virtual environment (not committed)
Setup and how to run
Requirements: Python 3.9+, and a Kaggle account to download the data.

cd ~/ml-projects/titanic
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook titanic.ipynb
Download train.csv and test.csv from the competition's Data tab into data/, then run the notebook top to bottom (Kernel → Restart & Run All).

Workflow: the big picture
 ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
 │  1. RAW DATA │ ──► │   2. EXPLORE │ ──► │ 3. FEATURE       │
 │  train.csv   │     │   (EDA)      │     │    ENGINEERING   │
 │  test.csv    │     │  who lived?  │     │  text → numbers  │
 └──────────────┘     └──────────────┘     └────────┬─────────┘
                                                    │
                                                    ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
 │ 6. TRAIN     │ ◄── │  5. SCALE    │ ◄── │ 4. SPLIT         │
 │ gradient     │     │  z-score     │     │ 80% train        │
 │ descent      │     │  (train μ,σ) │     │ 20% validation   │
 └──────┬───────┘     └──────────────┘     └──────────────────┘
        │
        ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
 │ 7. EVALUATE  │ ──► │  8. TUNE     │ ──► │ 9. FINAL MODEL   │
 │ train vs val │     │  α, λ,       │     │ retrain on all   │
 │ accuracy     │     │  features    │     │ 891 → predict    │
 └──────────────┘     └──────┬───────┘     │ test → Kaggle    │
        ▲                    │             └──────────────────┘
        └────────────────────┘
          repeat until validation
          accuracy stops improving
In one sentence: clean the data into numbers → hold some back for checking → teach the model with gradient descent → measure honestly on the held-back data → tune → retrain on everything → predict the unknown passengers.

Workflow: stage by stage
Stage 1: Raw data
Column	Example	Notes
Survived	0 / 1	The label y (train only)
Pclass	1, 2, 3	Ticket class
Name	"Braund, Mr. Owen Harris"	Contains a useful title
Sex	male / female	
Age	22	177 missing in train
SibSp, Parch	1, 0	Siblings/spouses and parents/children aboard
Fare	7.25	Very skewed
Cabin	C85	687 of 891 missing
Embarked	S / C / Q	2 missing
Stage 2: Explore (EDA)
The goal is to find out what drives survival before building anything.

Finding	Survival rate
Everyone	38%
Women vs men	74% vs 19%
1st vs 2nd vs 3rd class	63% vs 47% vs 24%
→ Sex and class are the strongest signals, so a good model must use them. These findings also give baselines to beat (61.6% and 78.7%).

Stage 3: Feature engineering (text → numbers)
A model can only do math on numbers, so every useful column is converted:

 RAW COLUMN            TRANSFORM                         MODEL FEATURE(S)
 ───────────           ─────────                         ────────────────
 Sex          ──►  female = 1, male = 0           ──►  Sex
 Pclass       ──►  one-hot (1st = default)        ──►  Pclass_2, Pclass_3
 Name         ──►  extract title → one-hot        ──►  Title_Mrs, Title_Miss,
                   (Mr = default)                      Title_Master, Title_Rare
 Age          ──►  fill missing with median        ──►  Age
                   age for that title
 Fare         ──►  log(1 + fare)                   ──►  LogFare
 SibSp+Parch  ──►  + 1                             ──►  FamilySize, IsAlone
 Cabin        ──►  is it missing?                  ──►  HasCabin
 Embarked     ──►  fill "S", one-hot (C = default) ──►  Embarked_Q, Embarked_S
 Ticket, PassengerId                               ──►  (dropped)
                                                        ─────────────
                                                        14 features
Why one-hot? Class 3 isn't "three times" class 1. Separate 0/1 columns let the model learn a separate effect for each category.

Stage 4: Train/validation split
           891 labeled passengers
 ┌─────────────────────────────────────────┐
 │████████████████████████████████│░░░░░░░░│
 └─────────────────────────────────────────┘
   712 TRAIN (80%)                  179 VAL (20%)
   model learns w, b here           only used to CHECK the model
If you only check accuracy on the data the model trained on, you can't see overfitting.
stratify keeps the 38% survival rate in both parts.
Stage 5: Feature scaling (z-score)
x_scaled = (x − μ) / σ        μ, σ computed on TRAIN only
Without scaling, Age (0–80) and IsAlone (0–1) have very different ranges, so gradient descent zig-zags and runs slowly.
The same μ and σ are reused on validation and test data. Computing them from validation or test data would leak information.
 TRAIN ──► compute μ, σ ──┬──► scale TRAIN
                          ├──► scale VALIDATION
                          └──► scale TEST (Kaggle)
Stage 6: Train (gradient descent)
See How the model learns below.

Stage 7: Evaluate
accuracy = (# correct predictions) / (# passengers)
Train acc	Val acc	Diagnosis
low	low	Underfitting (high bias): model too simple, or λ too big
high	much lower	Overfitting (high variance): model too complex, or λ too small
good	close to train	Just right ✅
Stage 8: Tune
Adjust α (learning rate), λ (regularization) and the feature set, and go back to Stage 6. Keep whatever gives the best validation accuracy.

Stage 9: Final model and submission
Retrain on all 891 labeled rows with the chosen α and λ (more data, a better model).
Apply the same feature steps and scaling to test.csv.
Predict 0/1 for the 418 test passengers → submission.csv → upload to Kaggle.
How the model makes one prediction
Logistic regression works in three steps:

 passenger features x                       weights w, bias b (learned)
 [Sex, Age, LogFare, …] ─────────┐   ┌───── [w₁, w₂, w₃, …], b
                                 ▼   ▼
                       ┌──────────────────────┐
  STEP 1: score        │  z = w · x + b       │   any number: −∞ … +∞
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
  STEP 2: probability  │  f = 1 / (1 + e^(−z))│   squashed to 0 … 1
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
  STEP 3: decision     │  f ≥ 0.5 → survived  │   0 or 1
                       │  f < 0.5 → died      │
                       └──────────────────────┘
Worked example (illustrative numbers)
The weights below are made up to show the mechanics. Your trained model will have its own values. Each "contribution" is wⱼ × xⱼ using the scaled feature.

Feature contribution (wⱼ·xⱼ)	👩 1st-class woman, 30	👨 3rd-class man, 25
Sex	+1.2	−0.8
Pclass_3	+0.5	−0.7
Title (Mrs vs Mr)	+0.3	−0.2
LogFare	+0.3	−0.2
Age	−0.1	0.0
Everything else	+0.1	−0.1
Sum of contributions	+2.3	−2.0
Bias b	−0.6	−0.6
z = w·x + b	+1.7	−2.6
f = sigmoid(z)	0.85 (85%)	0.07 (7%)
Prediction	Survived (1)	Died (0)
Reading it: each feature pushes the score up (toward survival) or down. The sigmoid turns the total score into a probability, and the 0.5 threshold turns that into a yes/no.

How the model learns (training loop)
The model starts knowing nothing (all weights 0) and improves step by step:

 ┌───────────────────────────────────────────────────────────────┐
 │ START: w = [0, 0, …, 0], b = 0                                 │
 │        every prediction = 0.5, cost J = 0.693                  │
 └───────────────────────────────┬───────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ 1. PREDICT    f = sigmoid(X·w + b)        for all 712 rows     │
 │ 2. MEASURE    J = −(1/m) Σ[y·log f + (1−y)·log(1−f)]           │
 │                   + (λ/2m) Σ wⱼ²                               │
 │ 3. GRADIENT   error = f − y                                    │
 │               ∂J/∂w = (1/m) Xᵀ·error + (λ/m)·w                 │
 │               ∂J/∂b = (1/m) Σ error                            │
 │ 4. UPDATE     w = w − α·∂J/∂w      (simultaneously)            │
 │               b = b − α·∂J/∂b                                  │
 └───────────────────────────────┬───────────────────────────────┘
                                 ▼
                     ┌───────────────────────┐   no
                     │ J stopped decreasing? │ ─────► back to step 1
                     └───────────┬───────────┘
                                 │ yes
                                 ▼
                     ┌───────────────────────┐
                     │ DONE: learned w, b    │
                     └───────────────────────┘
What each piece does
Piece	Plain-English meaning
Cost J	"How wrong are we on average?" Confidently wrong predictions cost the most.
error = f − y	For each passenger: predicted probability minus the truth.
∂J/∂w	For each feature: "if I nudge this weight up, does the cost go up or down, and by how much?"
α (learning rate)	Size of each step downhill.
λ (regularization)	Keeps weights small so the model doesn't memorize the training set.
Minus sign	Step opposite the slope, i.e. downhill.
What a healthy learning curve looks like
 cost J
 0.69 ┤●
      │ ●
      │  ●●
 0.55 ┤    ●●●
      │       ●●●●
 0.45 ┤           ●●●●●●●●●●●●●●●●●●     ← flat = converged
      └──────────────────────────────── iterations
       0     500    1000   1500   2000
Goes down every iteration → α is OK and the code is correct.
Goes up or bounces → α is too big, or there's a bug (e.g. + instead of −).
Barely moves → α is too small.
Tuning: choosing α and λ
Learning rate α
Try values that each go up about 3×: 0.001 → 0.003 → 0.01 → 0.03 → 0.1 → 0.3 → 1. Plot a learning curve for each and pick the largest α that still decreases J smoothly.

Regularization λ
 accuracy
     │   train ●───●───●──●
     │              ╲        ●
     │   val    ●──●──●╲       ●
     │        ╱         ╲●       ●
     │      ●              ●
     └──────┬───────┬───────┬────── λ
           0      best     100
       overfit    just     underfit
       (gap)      right    (both low)
Pick the λ with the best validation accuracy.

Key takeaways
Feature engineering mattered more than the algorithm. Turning Name into a title and SibSp + Parch into family size gave the model information it couldn't find on its own.
Validation data is essential. Without it, the overfit model looks better than it really is.
Scaling with training statistics only avoids data leakage.
Regularization lets you keep the extra features without overfitting.
My NumPy implementation matched scikit-learn's LogisticRegression (which uses C = 1/λ), confirming the math is right.
Next steps
 Try other features: deck letter from Cabin, ticket-group size, age bands
 Use k-fold cross-validation instead of a single split (Course 2)
 Compare with decision trees and random forests (Course 2)
 Build a "Would I have survived?" Streamlit app
Acknowledgements
Machine Learning Specialization by Andrew Ng, DeepLearning.AI and Stanford Online: the theory behind this project.
Kaggle: the Titanic - Machine Learning from Disaster competition and dataset.
