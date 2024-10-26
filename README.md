
# Anchors vs LIME for Text Classification

This repository contains the official implementation of the paper ["Comparing Feature Importance and Rule Extraction for Interpretability on Text Data"](https://arxiv.org/abs/2207.01420), XAIE @ ICPR 2022.

We show that using different methods can lead to unexpectedly different explanations, even when applied to simple models where we would expect qualitative consistency.

## Dependencies

The required Python packages are listed in `requirements.txt`.

Install the dependencies using the following commands:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the Experiments

```bash
git clone https://github.com/gianluigilopardo/anchors_vs_lime_text
python main.py
```

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@InProceedings{lopardo2022comparing, 
  title={Comparing Feature Importance and Rule Extraction for Interpretability on Text Data}, 
  author={Lopardo, Gianluigi and Garreau, Damien}, 
  year={2022}, 
  booktitle={International Conference on Pattern Recognition, 2-nd Workshop on Explainable and Ethical AI}
}
```
