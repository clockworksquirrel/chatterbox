import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Monkey patch torch.load to handle device mapping
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    """
    Loads a PyTorch object from a file, defaulting to CPU mapping if no device is specified.
    
    Parameters:
        f: The file-like object or file path to load from.
        map_location: The device to map the loaded tensors to. Defaults to 'cpu' if not provided.
    
    Returns:
        The deserialized PyTorch object with tensors mapped to the specified device.
    """
    if map_location is None:
        # Default to CPU for compatibility
        map_location = 'cpu'
    return original_torch_load(f, map_location=map_location, **kwargs)

torch.load = patched_torch_load

# Detect device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)
text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day."

# To synthesize with a different voice, uncomment the line below and specify the audio prompt
# AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(
    text, 
    # audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=2.0,
    cfg_weight=0.5
    )
ta.save("test-2.wav", wav, model.sr)
print("Audio saved to test-2.wav")
