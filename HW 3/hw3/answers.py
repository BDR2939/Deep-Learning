r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 128
    hypers["seq_len"] = 256
    hypers["h_dim"] = 512
    hypers["n_layers"] = 3
    hypers["dropout"] = 0.2
    hypers["learn_rate"] = 0.001
    hypers["lr_sched_factor"] = 0.2
    hypers["lr_sched_patience"] = 1
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "Hello to you, my name is "
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
In general, splitting the corpus to sequences provides better performance due to multiple factors. First, it allows parallel processing during training and significantly reduces memory requirements. Second, it helps the model to generalize better because it is forced to learn representations that are not reliant on the entire context of the text. This in turn helps the model to deal better with many NLP tasks such as machine translation which require the context of a word or phrase within a sentence.
"""

part1_q2 = r"""
During training, RNNs learn to update and refine their hidden states based on input sequences. This process allows the network to capture short-term dependencies within the sequence. However, the hidden state of an RNN is not strictly limited to information from a single sequence length. It retains information from previous time steps, which can potentially influence the generated text.
"""

part1_q3 = r"""
While shuffling batches is common in tasks like image classification where the order of samples does not matter, it is not suitable for text generation tasks due to the sequential nature and dependencies present in the data.
"""

part1_q4 = r"""
1. Lowering the temperature for sampling in text generation is a technique used to control the randomness of the generated output.
2. When the temperature is very high, the sampling process becomes more random. The model assigns more similar probabilities to a wider range of words, and allows for greater variance in the generated text. 
3. When the temperature is very low, the sampling becomes more deterministic. The model tends to select the most probable word or character at each step.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0,
        z_dim=0,
        x_sigma2=0,
        learn_rate=0.0,
        betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 64
    hypers["h_dim"] = 512
    hypers["z_dim"] = 32
    hypers["x_sigma2"] = 0.001
    hypers["learn_rate"] = 0.0001
    hypers["betas"] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The $\sigma^2$ parameter is a 

"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim=0,
        num_heads=0,
        num_layers=0,
        hidden_dim=0,
        window_size=0,
        droupout=0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
