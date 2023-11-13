from circlegen_functions import generate_json_history


if __name__=="__main__":
    phrase = "il risotto al tartufo è buono"
    generate_json_history(phrase, n_steps=100, fw_weight=.8, bw_weight=.5, beta=2.)
