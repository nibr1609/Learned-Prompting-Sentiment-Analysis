import torch as torch
from os.path import dirname
from transformers import pipeline


def make_pipeline(
    model_dir: str,
    pipeline_type: str = "text-generation",
    override_device: str = None,
):
    model_dir = (
        model_dir if model_dir.startswith("/") else dirname(__file__) + "/" + model_dir
    )

    device = (
        override_device
        or (torch.cuda.is_available() and "cuda")
        or (torch.mps.is_available() and "mps")
        or "cpu"
    )
    print(f"Using device: {device}")

    return pipeline(
        pipeline_type,
        model=model_dir,
        device=device,
        torch_dtype=torch.bfloat16,
    )


messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an expert in recognizin people's sentiments. Reply only with one word: positive, negative, or neutral.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "I FUCKING HATE THIS MOVIE! it sucks so bad... is what i would say if i was a loser. But actually, I love it!",
            },
        ],
    },
]
