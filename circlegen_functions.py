from modules import MinGPT,MinGPT_Trainer
from modules import get_tokenizer
import torch, torch.nn.functional as F, os, json
from pathlib import Path
from tqdm import tqdm
from parameters import *

states_dir = os.path.join(os.path.dirname(__file__),'states')
forward = os.path.join(states_dir,f'{MODEL_NAME}.state')
backward = os.path.join(states_dir,f'{MODEL_NAME}_backwards.state')

model_paths_by_orient = dict(fw=forward,bw=backward)

models = dict(fw=None,bw=None)
tokenizer = get_tokenizer(m_name='gpt2')
DEVICE = 'cuda:0'



filepath = Path(__file__).parent.as_posix()

def load_models(model_name, device='cpu'):
    """
        load models (bw,fw) from state, and its associated tokenizers.

        args :
        model_name : str, name of the model, as in states/<model_name>.state
                    Tokenizer folder should be states/<model_name>_tokenizer
        device : str, device to load the models on
    """
    tok_name = model_name.split('_')[0]
    states_dir = os.path.join(os.path.dirname(__file__),'states')

    tokenizer = get_tokenizer(m_path=os.path.join(states_dir,f'{tok_name}_tokenizer'),m_name=tok_name)

    forward = os.path.join(states_dir,f'{model_name}.state')
    backward = os.path.join(states_dir,f'{model_name}_backwards.state')
    model_paths_by_orient = dict(fw=forward,bw=backward)

    models = dict(fw=None,bw=None)
    # LOAD MODELS
    for orient, path in model_paths_by_orient.items():
        name, config, weights = MinGPT_Trainer.model_config_from_state(path,device=device)
        assert name=='MinGPT', 'For now only works with MinGPT models'

        models[orient] = MinGPT(**config).to(device).eval()
        models[orient].load_state_dict(weights,strict=True)
        models[orient].eval()
    return tokenizer, models

def roll_tokens(steps,tokens_by_orient):
    """
        Formerly in place, not anymore. If EFFICIENCY PROBLEMS, revisit.
    """
    return {'fw' : tokens_by_orient['fw'].roll(shifts=steps,dims=-1),
            'bw' :tokens_by_orient['bw'].roll(shifts=-steps,dims=-1)}

def resample_last_token(tokens_by_orient,models,fw_weight=0.5,bw_weight=0.5,beta=0.5):
    """
        Resample the last token of the phrase, given the weighted probabilyt of fw and bw
    """
    # Get logits for each orientation
    logits = dict(fw=None,bw=None)
    for orient, model in models.items():
        logits[orient] = model(tokens_by_orient[orient])[:,-2,:] # Select the next to last logit
    
    mean_logit = (fw_weight*logits['fw']+bw_weight*logits['bw'])/(fw_weight+bw_weight)
    mean_proba = F.softmax(mean_logit*beta,dim=-1)

    new_token = torch.multinomial(mean_proba,1)[0,0] # long number
    tokens_by_orient['fw'][0,-1] = new_token # (1, T) replaced last token
    tokens_by_orient['bw'][0,-1] = new_token # (1, T) replaced last token

def resample_worst_token(tokens_by_orient,models,fw_weight=0.5,bw_weight=0.5,beta=0.5):
    """
        Resamples the worst token, when evaluated fw and bw in all possible orientations.
        Does not work yet
    """
    rotated_tokens = dict(fw=None,bw=None)
    # Prepare the batch of all possible orientations
    _,T = tokens_by_orient['fw'].shape
    rotated_shape = (T,T)

    indices_fw = (torch.arange(T)[None,:] - torch.arange(T)[:,None]) % T
    indices_bw = (torch.arange(T)[None,:] + torch.arange(T)[:,None]) % T
    # Careful, this is a VIEW ! data is not duplicated

    rotated_tokens['fw'] = tokens_by_orient['fw'][:,indices_fw] # (T,T) 
    rotated_tokens['bw'] = tokens_by_orient['bw'][:,indices_bw] # (T,T)
    print('rotated : \n', rotated_tokens)


@torch.no_grad()
def generate_json_history(phrase, n_steps, model_name, fw_weight=.5, bw_weight=.5, beta=2.,device='cpu'):
    """
        Generate json file containing the history of the circle text generation
    """
    tokenizer, models = load_models(model_name,device=device)

    os.makedirs('json_history',exist_ok=True)
    numfiles = len(os.listdir('json_history'))
    jsonfile_store = os.path.join(filepath,'json_history',f'circle-dynamics_{numfiles}.json')
    jsonfile_draw = os.path.join(filepath,'circle_draw','circle-dynamics.json')
    jsonfull = os.path.join(filepath,'json_history',f'circle-dynamics-full.json') # FULL HISTORY FOR DEBUG

    tokened_phrase = tokenizer.tokenize(phrase).to(device) # (1, T)
    _,T = tokened_phrase.shape
    # Following contains forward and backward phrase, aligned s.t. the last token is the same
    tokens_by_orient= dict(fw=tokened_phrase,bw=torch.roll(torch.flip(tokened_phrase,dims=[-1]),shifts=-1,dims=-1))
    output_tokens = tokens_by_orient['fw']
    phrase_history = {'init_string':phrase,'edits':[]}
    full_phrase = [tokenizer.detokenize(output_tokens[0,:])]

    for n in tqdm(range(n_steps)):
        # Resample last token
        changed_token_length = len(tokenizer.detokenize(tokens_by_orient['fw'][:,-1]))

        resample_last_token(tokens_by_orient,models,fw_weight,bw_weight,beta)

        output_tokens = tokens_by_orient['fw'].roll(shifts=-(n%T),dims=-1) # Aligned with previous
        changed_token_location = (T-1-n)%T 
        changed_token_phrase_location = len(tokenizer.detokenize(output_tokens[:,:changed_token_location]))
        new_token = tokenizer.detokenize(output_tokens[:,changed_token_location])


        
        phrase_history['edits'].append({'loc':changed_token_phrase_location,'cut_len':changed_token_length,'token':new_token})
        full_phrase.append(tokenizer.detokenize(output_tokens))
        # Roll by 1 before next step :
        tokens_by_orient=roll_tokens(1,tokens_by_orient)

    
    with open(jsonfile_store,'w', encoding='utf-8') as f:
        json.dump(phrase_history,f,indent=4)

    with open(jsonfile_draw,'w', encoding='utf-8') as f:
        json.dump(phrase_history,f,indent=4)
    # FULL HISTORY FOR DEBUG
    with open(jsonfull,'w',encoding='utf-8') as f:
        json.dump(full_phrase,f,indent=4) 

# resample_worst_token({'fw':torch.tensor([[1,2,3,4]]),'bw':torch.tensor([[3,2,1,4]])},None,None,None)
