import random
import re
from causal.causal_model import CausalModel, CounterfactualDataset
from neural.LM_units import TokenPosition, get_last_token_index
from tasks.task import Task

# Constants
NUM_RANGE = 20  # x, y in range(20)
PROMPT_TEMPLATE = "<|endoftext|>, Yes, <x> + <y> = "


# Causal Model Functions
def compute_result(x, y):
    """Compute result = x + y"""
    return x + y


def format_prompt(*args):
    """Fill in the template with actual x, y values."""
    template, x, y = args[0], args[1], args[2]
    filled_template = template.replace("<x>", str(x)).replace("<y>", str(y))
    return filled_template


def format_output(result):
    """Format the output as a string"""
    return str(result)


# Define causal model structure
variables = ["template", "x", "y", "result", "raw_input", "raw_output"]

values = {
    "template": [PROMPT_TEMPLATE],
    "x": range(1, NUM_RANGE),
    "y": range(1, NUM_RANGE),
    "result": range(2, NUM_RANGE * 2 - 1),  # x + y can be up to 38
    "raw_input": None,
    "raw_output": None,
}

parents = {
    "template": [],
    "x": [],
    "y": [],
    "result": ["x", "y"],
    "raw_input": ["template", "x", "y"],
    "raw_output": ["result"],
}

mechanisms = {
    "template": lambda: PROMPT_TEMPLATE,
    "x": lambda: random.randint(1, NUM_RANGE - 1),
    "y": lambda: random.randint(1, NUM_RANGE - 1),
    "result": compute_result,
    "raw_input": format_prompt,
    "raw_output": format_output,
}

addition_causal_model = CausalModel(
    variables,
    values,
    parents,
    mechanisms,
    id="two_number_addition"
)


# Dataset Sampler Functions
def sample_addition_problem():
    """Sample a random addition problem."""
    input_sample = addition_causal_model.sample_input()
    input_sample["raw_input"] = addition_causal_model.run_forward(input_sample)["raw_input"]
    return input_sample


def intervene_on_x():
    """
    Generate a counterfactual where we intervene on x.
    This changes the final result.
    """
    input_sample = sample_addition_problem()
    counterfactual = input_sample.copy()
    del counterfactual["raw_input"]

    # Change x to a different value
    current_x = input_sample["x"]
    possible_values = [i for i in range(1, NUM_RANGE) if i != current_x]
    counterfactual["x"] = random.choice(possible_values)

    # Regenerate raw_input for both samples
    input_sample["raw_input"] = addition_causal_model.run_forward(input_sample)["raw_input"]
    counterfactual["raw_input"] = addition_causal_model.run_forward(counterfactual)["raw_input"]

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def intervene_on_y():
    """
    Generate a counterfactual where we intervene on y.
    This changes the final result.
    """
    input_sample = sample_addition_problem()
    counterfactual = input_sample.copy()
    del counterfactual["raw_input"]

    # Change y to a different value
    current_y = input_sample["y"]
    possible_values = [i for i in range(1, NUM_RANGE) if i != current_y]
    counterfactual["y"] = random.choice(possible_values)

    # Regenerate raw_input for both samples
    input_sample["raw_input"] = addition_causal_model.run_forward(input_sample)["raw_input"]
    counterfactual["raw_input"] = addition_causal_model.run_forward(counterfactual)["raw_input"]

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def random_counterfactual():
    """
    Generate a completely random counterfactual by sampling two independent inputs.
    """
    input_sample = sample_addition_problem()
    counterfactual = sample_addition_problem()
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def distinct_counterfactual():
    """
    Generate a counterfactual where we intervene on a distinct variable from the original input.
    """
    input_sample = sample_addition_problem()
    while True:
        counterfactual = sample_addition_problem()
        x0, y0 = input_sample["x"], input_sample["y"]
        x1, y1 = counterfactual["x"], counterfactual["y"]
        if x0 != x1 and y0 != y1 and x0+y0 != x1+y1:
            break
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


# Token Position Functions
def get_summand_index(input_sample, pipeline, position):
    """
    Find the index of a summand (x or y) in the prompt.

    Args:
        input_sample (Dict): The input dictionary to a causal model
        pipeline: The tokenizer pipeline
        position (int): 0 for x, 1 for y

    Returns:
        list[int]: List containing the index of the summand token (last token of the number)
    """
    if position not in [0, 1]:
        raise ValueError(f"Position must be 0 or 1, got {position}")

    # Step 1: Load FULL prompt WITH padding (as normal)
    prompt = input_sample["raw_input"]
    tokenized_prompt_padded = list(pipeline.load(prompt)["input_ids"][0])
    pad_token_id = pipeline.tokenizer.pad_token_id

    # Step 2: Find where content starts (first non-padding token)
    content_start_idx = 0
    for i, token in enumerate(tokenized_prompt_padded):
        if token != pad_token_id:
            content_start_idx = i
            break
    
    # Step 3: Hard-coded token positions
    return [content_start_idx + 4 + 3 * position]


def create_summand_token_position(pipeline, position):
    """
    Create a TokenPosition for a summand (x or y).

    Args:
        pipeline: The tokenizer pipeline
        position (int): 0 for x, 1 for y

    Returns:
        TokenPosition: Token position for the specified summand
    """
    summand_names = ["x", "y"]
    return TokenPosition(
        lambda x: get_summand_index(x, pipeline, position),
        pipeline,
        id=summand_names[position]
    )


def create_last_token_position(pipeline):
    """Create a TokenPosition for the last token in the input."""
    return TokenPosition(
        lambda x: get_last_token_index(x, pipeline),
        pipeline,
        id="last_token"
    )

# def create_arb_token_position(pipeline, i):
#     return TokenPosition(
#         lambda x: [i],
#         pipeline,
#         id=f"arb_token_{i}"
#     )


# Task Definition
addition_task = Task(
    name="addition",
    causal_models={
        "two_number_addition": addition_causal_model,
    },
    dataset_generators={
        "intervene_on_x": intervene_on_x,
        "intervene_on_y": intervene_on_y,
        "random_counterfactual": random_counterfactual,
        "distinct_counterfactual": distinct_counterfactual,
    },
    token_positions={
        "x": lambda x: create_summand_token_position(x, 0),
        "y": lambda x: create_summand_token_position(x, 1),
        "last_token": create_last_token_position,
    }
)
