# WordLoop

## How to use :
Run circlegen.py to generate a json file with the evolution.
To modify the init phrase, do so directly in circlegen.py.

To modify which tokenizer/models are used, specify their name in circlegen.py.
For example, if you have, inside the 'states' folder : `states/en_med.state` and `states/en_med_backwards.states`.
You should put `model_name = 'en'`. There should also be a `en_tokenizer` folder containing the appropriate tokenizer.