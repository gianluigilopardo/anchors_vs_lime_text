# Comparing Anchors and LIME for Text Data

## Requirements

Some non-standard packages need to be installed:
 - anchor (repo from the authors)
 ```
 pip install anchor-exp
 ```
 - spacy and a pretrained model:
 ```
 pip install spacy
 python -m spacy download en_core_web_sm
 ```

## Use

To generate figures for the experiments, simply do 
```
git clone https://github.com/gianluigilopardo/anchors_vs_lime_text
python main.py
```
