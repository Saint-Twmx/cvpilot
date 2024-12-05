import torch
import numpy as np
import json
import math
import os
def transf(head, data: torch.tensor):
    scale = torch.tensor(head["spacing"])
    shift = torch.tensor(head["origin"][::-1])
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.cpu()
    data = data * scale + shift
    return np.array(data[:, [2, 1, 0]]).tolist()

def post_processing_measure(
    head,
    measure_result: dict,
):
    filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
            "measurement.json"
        )

    with open(filepath, 'r', encoding='utf-8') as file:
        mj = json.load(file)


    save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
            "measurement.json"
        )
    with open(save_path, 'w') as file:
        json.dump(mj, file)