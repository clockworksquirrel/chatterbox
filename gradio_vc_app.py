import random
import numpy as np
import torch
import gradio as gr
from chatterbox.vc import ChatterboxVC
import time
import logging
import sys
from optimum.bettertransformer import BetterTransformer

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model variable
GLOBAL_MODEL = None

# Device detection
if torch.backends.mps.is_available():
    DEVICE = "mps"
    logger.info("🚀 Apple Silicon MPS backend is available, will use GPU after CPU loading.")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info("🚀 CUDA GPU detected, will use GPU after CPU loading")
else:
    DEVICE = "cpu"
    logger.info("🚀 No GPU detected, running on CPU only")


def set_seed(seed: int):
    """
    Set the random seed for PyTorch, CUDA, Python's random module, and NumPy to ensure reproducible results.
    
    Parameters:
        seed (int): The seed value to use for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def warmup_model(model):
    """
    Performs a warm-up pass on the model by converting a dummy silent audio file to ensure all kernels are compiled and the model is ready for inference.
    
    Returns:
        bool: True if the warm-up succeeds and valid audio is generated, False otherwise.
    """
    logger.info("🔥 Starting model warm-up...")
    tmp_path = None
    try:
        start_time = time.time()
        
        # Create a dummy audio for warm-up (1 second of silence)
        sample_rate = 16000
        duration = 1.0
        dummy_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        # Save to temporary file
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, dummy_audio, sample_rate)
            tmp_path = tmp_file.name
        
        # Run voice conversion
        result = model.generate(tmp_path, target_voice_path=tmp_path)
        
        if result is None or result.shape[-1] == 0:
            logger.error("❌ Warm-up failed: No audio generated")
            return False
            
        duration = result.shape[-1] / model.sr
        end_time = time.time()
        logger.info(f"✅ Model warm-up completed in {end_time - start_time:.2f} seconds")
        logger.info(f"   Generated {duration:.2f} seconds of audio")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model warm-up failed: {e}", exc_info=True)
        return False
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            import os
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")


def transfer_model_to_gpu(model, device):
    """
    Move all submodules of a ChatterboxVC model instance to the specified GPU device.
    
    Parameters:
        model: The ChatterboxVC model instance to transfer.
        device (str): Target device identifier ('cuda' or 'mps').
    
    Returns:
        The model with all components moved to the specified device.
    
    Raises:
        Exception: If any component fails to transfer to the target device.
    """
    logger.info(f"📤 Transferring model components from CPU to {device}...")
    start_time = time.time()
    
    try:
        # Transfer S3Gen component
        logger.info("   - Transferring S3Gen model...")
        model.s3gen = model.s3gen.to(device)
        
        # Update device attribute
        model.device = device
        
        # Transfer ref_dict if it exists
        if model.ref_dict is not None:
            logger.info("   - Transferring reference embeddings...")
            model.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in model.ref_dict.items()
            }
        
        # Synchronize to ensure transfers complete
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
            
        end_time = time.time()
        logger.info(f"✅ Model components successfully transferred to {device} in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Failed to transfer model to GPU: {e}", exc_info=True)
        raise
        
    return model


def load_and_prepare_model():
    """
    Initializes and prepares the ChatterboxVC model for inference.
    
    Loads the model on CPU, transfers it to GPU if available, applies BetterTransformer optimizations to key components, and performs a warm-up conversion to ensure readiness. Raises a RuntimeError if the warm-up fails.
    
    Returns:
        model: The fully initialized and ready-to-use ChatterboxVC model instance.
    """
    logger.info("=" * 60)
    logger.info("🚀 INITIALIZING CHATTERBOX VC MODEL")
    logger.info("=" * 60)
    
    # Step 1: Load model on CPU
    logger.info("📥 Step 1: Loading model on CPU...")
    start_time = time.time()
    model = ChatterboxVC.from_pretrained("cpu")
    logger.info(f"✅ Model loaded on CPU in {time.time() - start_time:.2f} seconds")
    
    # Step 2: Transfer to GPU if available
    if DEVICE != "cpu":
        logger.info(f"📤 Step 2: Transferring model to {DEVICE}...")
        model = transfer_model_to_gpu(model, DEVICE)
    else:
        logger.info("ℹ️  Step 2: Skipping GPU transfer (CPU-only mode)")
    
    # Step 3: Apply BetterTransformer optimization
    # logger.info("⚡ Step 3: Applying BetterTransformer optimization...")
    # try:
    #     opt_start = time.time()
    #     model.s3gen = BetterTransformer.transform(model.s3gen)
    #     logger.info(f"✅ BetterTransformer applied in {time.time() - opt_start:.2f} seconds")
    # except Exception as e:
    #     logger.warning(f"⚠️  Could not apply BetterTransformer: {e}")
    
    # Step 4: Warm up the model
    logger.info("🔥 Step 4: Warming up the model...")
    warmup_success = warmup_model(model)
    
    if not warmup_success:
        logger.error("❌ Model warm-up failed!")
        raise RuntimeError("Model warm-up failed. Please check the logs.")
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"✅ MODEL READY! Total initialization time: {total_time:.2f} seconds")
    logger.info("=" * 60)
    
    return model


def convert(src_wav_path, ref_wav_path, seed_num):
    """
    Performs voice conversion using the globally loaded ChatterboxVC model.
    
    Parameters:
        src_wav_path (str): Path to the source audio file to be converted.
        ref_wav_path (str): Path to the reference audio file providing the target voice characteristics.
        seed_num (int): Random seed for reproducibility; if 0, no seed is set.
    
    Returns:
        tuple: The result of the voice conversion, typically containing the sample rate and the generated audio array.
    """
    if GLOBAL_MODEL is None:
        raise RuntimeError("Model not initialized!")

    if seed_num != 0:
        set_seed(int(seed_num))

    logger.info("🎤 Converting voice...")
    logger.info(f"   Source: {src_wav_path}")
    logger.info(f"   Reference: {ref_wav_path}")
    start_time = time.time()

    # ChatterboxVC only uses the audio and target_voice_path
    result = GLOBAL_MODEL.generate(
        src_wav_path,
        target_voice_path=ref_wav_path,
    )

    end_time = time.time()
    if result is not None:
        duration = result.shape[-1] / GLOBAL_MODEL.sr
        total_time = end_time - start_time
        rtf = total_time / duration
        logger.info(f"✅ Generated {duration:.2f}s of audio in {total_time:.2f}s (RTF: {rtf:.2f})")
        # Return in the expected format (sample_rate, audio_array)
        return (GLOBAL_MODEL.sr, result.squeeze(0).cpu().numpy())
    
    logger.error("❌ Voice conversion failed")
    return None


# Initialize model before creating the interface
logger.info("🚀 Starting Chatterbox VC Application...")
try:
    GLOBAL_MODEL = load_and_prepare_model()
except Exception as e:
    logger.error(f"Failed to initialize model: {e}", exc_info=True)
    raise

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Chatterbox Voice Conversion")
    gr.Markdown("Model loaded and warmed up. Ready for fast inference!")
    gr.Markdown("**Note:** Voice Conversion only uses source and reference audio. Generation parameters like steps, temperature, etc. are not applicable for audio-to-audio conversion.")
    
    with gr.Row():
        with gr.Column():
            src_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Source Audio File")
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File")
            seed_num = gr.Number(value=0, label="Random seed (0 for random)")

            run_btn = gr.Button("Convert", variant="primary")

        with gr.Column():
            gr.Markdown("### 🎵 Converted Audio")
            gr.Markdown("*Click the download button (⬇️) in the audio player to save the converted audio file*")
            audio_output = gr.Audio(
                label="Output Audio",
                type="numpy",
                show_download_button=True,
                format="wav"
            )
            
            # Add conversion info display
            conversion_info = gr.Markdown(visible=False)

    def convert_with_info(*args):
        """Wrapper to add conversion info display"""
        import time
        start_time = time.time()
        result = convert(*args)
        end_time = time.time()
        
        if result is not None:
            sr, audio = result
            duration = len(audio) / sr
            conversion_time = end_time - start_time
            rtf = conversion_time / duration
            
            info_text = f"""✅ **Conversion complete!**
- Audio duration: {duration:.2f} seconds
- Conversion time: {conversion_time:.2f} seconds
- Real-time factor (RTF): {rtf:.2f}x
- Sample rate: {sr:,} Hz
- Click the download button above to save"""
            
            return result, gr.update(value=info_text, visible=True)
        else:
            return None, gr.update(value="❌ Voice conversion failed", visible=True)

    run_btn.click(
        fn=convert_with_info,
        inputs=[
            src_wav,
            ref_wav,
            seed_num,
        ],
        outputs=[audio_output, conversion_info],
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=False)
