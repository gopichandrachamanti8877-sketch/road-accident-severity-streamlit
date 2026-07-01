
# Road Accident Severity Classification Frontend

This Streamlit app loads the trained CNN model and predicts accident severity from an uploaded image.

## Classes
- Minor Impact
- Substantial Impact
- Critical Impact

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `best_accident_severity_model.keras` is in the same folder as `app.py`.
