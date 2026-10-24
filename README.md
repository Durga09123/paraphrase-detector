# Paraphrase Detector (NLP Lab - Project 6)

Checks whether two sentences have the same meaning using
**WordNet synonym normalisation + TF-IDF + Cosine Similarity** (threshold = 0.30).

## Run locally in VS Code
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 (NLTK data downloads automatically on first run).

## Deploy on Vercel
1. Push this folder to a GitHub repository (see steps in chat).
2. Vercel -> Add New -> Project -> import the repo -> Deploy (no settings needed).

## Files
| File | Purpose |
|------|---------|
| app.py | Flask web app (replaces Streamlit, which Vercel cannot run) |
| detector.py | Pre-processing, TF-IDF, cosine similarity, paraphrase decision |
| templates/index.html | Web page |
| requirements.txt | Dependencies |
| vercel.json | Vercel settings |
