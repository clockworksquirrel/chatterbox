from pathlib import Path

import librosa
import torch
import perth
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen


REPO_ID = "ResembleAI/chatterbox"


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        ckpt_dir = Path(ckpt_dir)
        
        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
            
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        """
        Instantiate a ChatterboxVC model by downloading pretrained weights and reference conditionals from the Hugging Face hub.
        
        Parameters:
            device (str): The device identifier (e.g., 'cpu', 'cuda', 'mps') on which to load the model.
        
        Returns:
            ChatterboxVC: An instance of the voice conversion model initialized with pretrained parameters.
        """
        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
    ):
        """
        Convert input audio to the target voice using the pretrained speech generator and reference embeddings.
        
        Parameters:
            audio (str or torch.Tensor): Input audio as a file path or a waveform tensor.
            target_voice_path (str, optional): Path to a WAV file for the target voice reference. If not provided, precomputed reference embeddings must be set.
        
        Returns:
            torch.Tensor: The generated waveform with the target voice, watermarked and returned as a tensor with a batch dimension.
        """
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `prepare_conditionals` first or specify `target_voice_path`"

        with torch.inference_mode():
            if isinstance(audio, str):
                audio_16, _ = librosa.load(audio, sr=S3_SR)
                audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]
            elif isinstance(audio, torch.Tensor):
                # Validate tensor shape and move to device if needed
                if audio.dim() == 1:
                    audio_16 = audio.unsqueeze(0)
                else:
                    audio_16 = audio
                if audio_16.device != self.device:
                    audio_16 = audio_16.to(self.device)
            else:
                raise TypeError(f"audio must be a file path (str) or torch.Tensor, got {type(audio)}")

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)