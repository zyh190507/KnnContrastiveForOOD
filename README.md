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
**ATTENTION!! Incorrect filling due to carelessness**, In the BANKING 25% of Table1 in paper, the actual F1-IND result of SCL+LOF is **74.12**(**less than OUR 76.44**), not 79.28 in the paper. **Sorry for this carelessness**.

