import random
import re
from causal.causal_model import CausalModel, CounterfactualDataset
from neural.LM_units import TokenPosition, get_last_token_index
from tasks.task import Task

# Constants
NUM_RANGE = 20  # x, y, z in range(100)
PROMPT_TEMPLATE = "<|endoftext|>, Yes, (<x> + <y>) + <z> = "


# Causal Model Functions
def compute_intermediate(x, y):
    """Compute x' = x + y"""
    return x + y


def compute_final(x_plus_y, z):
    """Compute x'' = x' + z"""
    return x_plus_y + z


def format_prompt(*args):
    """Fill in the template with actual x, y, z values."""
    template, x, y, z = args[0], args[1], args[2], args[3]
    filled_template = template.replace("<x>", str(x)).replace("<y>", str(y)).replace("<z>", str(z))
    return filled_template


def format_output(result):
    """Format the output as a string"""
    return str(result)


# Define causal model structure
variables = ["template", "x", "y", "z", "x_plus_y", "result", "raw_input", "raw_output"]

values = {
    "template": [PROMPT_TEMPLATE],
    "x": range(1, NUM_RANGE),
    "y": range(1,NUM_RANGE),
    "z": range(1,NUM_RANGE),
    "x_plus_y": range(2, NUM_RANGE * 2 - 1),  # x + y can be up to 198
    "result": range(3, NUM_RANGE * 3 - 2),    # x + y + z can be up to 297
    "raw_input": None,
    "raw_output": None,
}

parents = {
    "template": [],
    "x": [],
    "y": [],
    "z": [],
    "x_plus_y": ["x", "y"],
    "result": ["x_plus_y", "z"],
    "raw_input": ["template", "x", "y", "z"],
    "raw_output": ["result"],
}

mechanisms = {
    "template": lambda: PROMPT_TEMPLATE,
    "x": lambda: random.randint(1, NUM_RANGE - 1),
    "y": lambda: random.randint(1, NUM_RANGE - 1),
    "z": lambda: random.randint(1, NUM_RANGE - 1),
    "x_plus_y": compute_intermediate,
    "result": compute_final,
    "raw_input": format_prompt,
    "raw_output": format_output,
}

addition_causal_model = CausalModel(
    variables,
    values,
    parents,
    mechanisms,
    id="three_number_addition"
)


# Dataset Sampler Functions
def sample_triple_add_problem():
    """Sample a random addition problem."""
    input_sample = addition_causal_model.sample_input()
    input_sample["raw_input"] = addition_causal_model.run_forward(input_sample)["raw_input"]
    return input_sample


def intervene_on_x():
    """
    Generate a counterfactual where we intervene on x.
    This changes the intermediate result x' and the final result x''.
    """
    input_sample = sample_triple_add_problem()
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
    This changes the intermediate result x' and the final result x''.
    """
    input_sample = sample_triple_add_problem()
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


def intervene_on_z():
    """
    Generate a counterfactual where we intervene on z.
    This changes only the final result x'' but not the intermediate x'.
    """
    input_sample = sample_triple_add_problem()
    counterfactual = input_sample.copy()
    del counterfactual["raw_input"]

    # Change z to a different value
    current_z = input_sample["z"]
    possible_values = [i for i in range(1, NUM_RANGE) if i != current_z]
    counterfactual["z"] = random.choice(possible_values)

    # Regenerate raw_input for both samples
    input_sample["raw_input"] = addition_causal_model.run_forward(input_sample)["raw_input"]
    counterfactual["raw_input"] = addition_causal_model.run_forward(counterfactual)["raw_input"]

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def intervene_on_intermediate():
    """
    Generate a counterfactual where we intervene on the intermediate variable x_plus_y.
    This directly sets x' to a different value, affecting the final result.
    """
    input_sample = sample_triple_add_problem()
    counterfactual = input_sample.copy()
    del counterfactual["raw_input"]

    # Change x_plus_y to a different value
    current_intermediate = addition_causal_model.run_forward(input_sample)["x_plus_y"]
    possible_values = [i for i in range(2, NUM_RANGE * 2 - 1) if i != current_intermediate]
    counterfactual["x_plus_y"] = random.choice(possible_values)

    # Regenerate raw_input for both samples
    input_sample["raw_input"] = addition_causal_model.run_forward(input_sample)["raw_input"]
    counterfactual["raw_input"] = addition_causal_model.run_forward(counterfactual)["raw_input"]

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def random_counterfactual():
    """
    Generate a completely random counterfactual by sampling two independent inputs.
    """
    input_sample = sample_triple_add_problem()
    counterfactual = sample_triple_add_problem()
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def distinct_counterfactual():
    """
    Generate a counterfactual where we intervene on a distinct variable from the original input.
    """
    input_sample = sample_triple_add_problem()
    while True:
        counterfactual = sample_triple_add_problem()
        x0, y0, z0 = input_sample["x"], input_sample["y"], input_sample["z"]
        x1, y1, z1 = counterfactual["x"], counterfactual["y"], counterfactual["z"]
        if x0 != x1 and y0 != y1 and z0 != z1 and x0+y0 != x1+y1 and x0+y0+z0 != x1+y1+z1:
            break
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


# Token Position Functions
def get_summand_index(input_sample, pipeline, position):
    """
    Find the index of a summand (x, y, or z) in the prompt.

    Args:
        input_sample (Dict): The input dictionary to a causal model
        pipeline: The tokenizer pipeline
        position (int): 0 for x, 1 for y, 2 for z

    Returns:
        list[int]: List containing the index of the summand token (last token of the number)
    """
    if position not in [0, 1, 2]:
        raise ValueError(f"Position must be 0, 1, or 2, got {position}")

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


def get_summand_index_plus_one(input_sample, pipeline, position):
    """
    Find the index one after a summand (x, y, or z) in the prompt.
    """
    return [get_summand_index(input_sample, pipeline, position)[0] + 1]


def get_summand_index_plus_two(input_sample, pipeline, position):
    return [get_summand_index(input_sample, pipeline, position)[0] + 2]


def get_first_token_index(input_sample, pipeline):
    """
    Find the index of the first token in the input.
    """
    # Step 1: Load FULL prompt WITH padding (as normal)
    prompt = input_sample["raw_input"]
    tokenized_prompt_padded = list(pipeline.load(prompt)["input_ids"][0])
    pad_token_id = pipeline.tokenizer.pad_token_id

    # Step 2: Find where content starts (first non-padding token)
    for i, token_id in enumerate(tokenized_prompt_padded):
        if token_id != pad_token_id:
            content_start_idx = i
            break
    
    return [content_start_idx]


def create_summand_token_position(pipeline, position):
    """
    Create a TokenPosition for a summand (x, y, or z).

    Args:
        pipeline: The tokenizer pipeline
        position (int): 0 for x, 1 for y, 2 for z

    Returns:
        TokenPosition: Token position for the specified summand
    """
    summand_names = ["x", "y", "z"]
    return TokenPosition(
        lambda x: get_summand_index(x, pipeline, position),
        pipeline,
        id=summand_names[position]
    )


def create_summand_token_position_plus_one(pipeline, position):
    """Create a TokenPosition for a summand (x, y, or z) plus one."""
    summand_names = ["x", "y", "z"]
    return TokenPosition(
        lambda x: get_summand_index_plus_one(x, pipeline, position),
        pipeline,
        id=summand_names[position] + "_plus_one"
    )


def create_summand_token_position_plus_two(pipeline, position):
    summand_names = ["x", "y", "z"]
    return TokenPosition(
        lambda x: get_summand_index_plus_two(x, pipeline, position),
        pipeline,
        id=summand_names[position] + "_plus_two"
    )


def create_last_token_position(pipeline):
    """Create a TokenPosition for the last token in the input."""
    return TokenPosition(
        lambda x: get_last_token_index(x, pipeline),
        pipeline,
        id="last_token"
    )


# Task Definition
triple_add_task = Task(
    name="addition",
    causal_models={
        "three_number_addition": addition_causal_model,
    },
    dataset_generators={
        "intervene_on_x": intervene_on_x,
        "intervene_on_y": intervene_on_y,
        "intervene_on_z": intervene_on_z,
        "intervene_on_intermediate": intervene_on_intermediate,
        "random_counterfactual": random_counterfactual,
        "distinct_counterfactual": distinct_counterfactual,
    },
    token_positions={
        "x": lambda x: create_summand_token_position(x, 0),
        "x_plus_one": lambda x: create_summand_token_position_plus_one(x, 0),
        "x_plus_two": lambda x: create_summand_token_position_plus_two(x, 0),
        "y": lambda x: create_summand_token_position(x, 1),
        "y_plus_one": lambda x: create_summand_token_position_plus_one(x, 1),
        "y_plus_two": lambda x: create_summand_token_position_plus_two(x, 1),
        "z": lambda x: create_summand_token_position(x, 2),
        "z_plus_one": lambda x: create_summand_token_position_plus_one(x, 2),
        "last_token": create_last_token_position,
    },
)
