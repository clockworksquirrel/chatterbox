import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Monkey patch torch.load to handle device mapping
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    """
    Loads a PyTorch object from a file, defaulting to CPU device mapping if no device is specified.
    
    Parameters:
        f: The file-like object or file path to load the object from.
        map_location: Optional device mapping for loaded tensors. Defaults to 'cpu' if not provided.
    
    Returns:
        The deserialized PyTorch object with tensors mapped to the specified device.
    """
    if map_location is None:
        # Default to CPU for compatibility
        map_location = 'cpu'
    return original_torch_load(f, map_location=map_location, **kwargs)

torch.load = patched_torch_load

# Automatically detect the best available device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
