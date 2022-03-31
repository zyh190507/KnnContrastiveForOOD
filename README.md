## Source code for 'KNN-Contrastive Learning for Out-of-Domain Intent Classofication'.

## Dependencies
### Use anaconda to create python environemnt:
[conda create --name python=3.6.12]

### Install all required libraries:
[pip install -r requirements.txt]

### Usage:
Run the experiments (for example banking_25):

[python run_main.py json/banking_25.json]

### Fix:
Due to the wrong filling, the F1-IND of SCL+LOF is **74.12 (OUR is 76.44) **, not 79.28 in the paper.

