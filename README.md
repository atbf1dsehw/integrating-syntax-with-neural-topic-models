# integrating-syntax-with-neural-topic-models
Code for integrating topics with neural topic models.

Additional installation for spacy tokenizer:

```python -m spacy download en_core_web_sm```

## Usage

Included are scripts for training the models in the paper.

Follow these instructions to train the models:
1. Make an environment with the requirements.txt file
2. set a root path where you want to store the data, models, and results.
3. run the respective scripts in the root directory of the repository (after setting the root path in all scripts)
4. We recommend following order of scripts:
    1. ```context_net.sh```
    2. ```etm.sh```
    3. ```etm_dirichlet.sh```
    4. ```dvae.sh```
    5. ```lda.sh```
    6. ```syconntm_pretrained.sh``` (include the root path for context net in the script)
    7. ```qualitative_assessment_pretrained.sh```
5. Run analysis notebook in src folder (provide paths wherever appropriate) to get the results and plots.
