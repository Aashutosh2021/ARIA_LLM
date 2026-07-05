"""
AIRA-LLM
Temperature Scaling
"""


def apply_temperature(logits, temperature: float = 1.0):
    """
    Scale logits by temperature.

    temperature < 1.0 -> sharper (more greedy)
    temperature > 1.0 -> flatter (more random)
    temperature == 0  -> caller should use greedy argmax instead
    """

    if temperature is None or temperature <= 0:
        return logits

    if temperature == 1.0:
        return logits

    return logits / temperature
