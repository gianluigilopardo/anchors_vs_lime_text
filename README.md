# Comparing Interpretability Methods for Text Data

Machine learning algorithms are used more and more often in critical tasks involving text data. 
The increase in model complexity has made understanding their decisions increasingly challenging, leading to the development of interpretability methods. 
Among local methods, two families have emerged: those computing importance scores for each features and those extracting logical simple rules. 
We show that different methodologies can produce different and sometimes unexpected results, even when applied to simple models for which we would expect them to coincide qualitatively. 

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
