# Plate: AI Workout Recommendation (Streamlit + scikit-learn)

# Author: Mohammed Aasik.a

github: https://github.com/mdaasik-tech
linkedin: https://www.linkedin.com/in/mohammed-aasik-257787433/


Personalised single-muscle workouts built from your curated `exercises.csv` (122 exercises, 7 muscle groups).
Transparent rules decide what is safe and suitable. Machine learning helps where it can be justified.

## Run
```bash
python -m pip install -r requirements.txt
streamlit run app.py
python -m pytest            # 28 tests: data, rules, ML gate, headless UI
```

## What is inside
| Page | What it does |
|---|---|
| Home | Animated intro, muscle groups, data-review notice |
| Create Workout | 5 steps: goal, level, muscle (+emphasis), gear and time, limits |
| My Workout | Plan with count-up stats, filter funnel, per-exercise "why", swaps, TXT/CSV download, ratings |
| ML Lab | Exercise map, similarity finder, feedback-model training with a validation gate |
| About & Safety | Method, data limits, privacy, medical disclaimer |

## Pipeline (follows your specification)
1. `data_loader.py` validates every column, allowed value, duplicate ID and `alternative_ids` reference.
2. `safety_filters.py` runs first: avoided exercises, protected body areas (shoulder, lower back, wrist read from `contraindication_notes`), no high-difficulty moves when pain, injury or a medical restriction is reported.
3. `recommender.py` filters by muscle, equipment, location and level, then ranks with the 35/20/20/15/10 weighted score. Every component is 0..1.
4. `workout_builder.py` composes the session with MMR diversification (relevance vs cosine similarity), fits the time budget, prescribes sets, reps and rest from your goal and intensity, and orders compound before isolation before isometric.
5. Swaps use reviewed `alternative_ids` first, then content-similar exercises, always inside the user's constraints.

## Machine learning, honestly
* **Content similarity (works now).** TF-IDF over muscles, pattern, equipment, type and difficulty, then cosine similarity. Used for swaps, near-duplicate avoidance and the exercise map. It needs no user data.
* **Feedback model (waits for data).** Logistic Regression or Gradient Boosting predicts a 4+ star rating. It is trained only on consented ratings, never on pain, injury or medical answers, and is evaluated with grouped cross-validation against the rule baseline. It activates only if it has 100+ ratings, a gain of at least 0.02 AUC that exceeds fold noise, and AUC of at least 0.60. It then contributes 20% of the score and can only re-order exercises that already passed safety.
* `data/feedback_SIMULATED.csv` and `scripts/simulate_feedback.py` exist to test the pipeline. Simulated data is never used to activate a model. On this file the gate correctly refuses the model, which is the expected result.

## Before you publish
* Every exercise is `reviewed=false`. Have a qualified coach review the catalogue and set `reviewed=true`.
* CSV feedback is for a single-user prototype. Streamlit Community Cloud may reset local files on restart, and a shared CSV is unsafe for concurrent writers. Move feedback to a database before a public launch.
* `.gitignore` keeps real feedback and trained models out of GitHub.

## Deploy
Push to GitHub, open https://share.streamlit.io, choose the repo, branch `main`, main file `app.py`.
The light theme is fixed in `.streamlit/config.toml` because the custom CSS uses fixed colours.
