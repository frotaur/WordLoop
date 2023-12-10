from circlegen_functions import generate_json_history


if __name__=="__main__":
    model_name = "en_med"
    phrase = "just wanna see if this works, man !"

    generate_json_history(phrase, n_steps=300, model_name=model_name,fw_weight=.8, bw_weight=.5, beta=2.,device='cuda')
