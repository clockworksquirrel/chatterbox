import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
import time
import re
from typing import List
import logging
import sys
from dataclasses import dataclass
from optimum.bettertransformer import BetterTransformer
from mps_fast_patch import optimize_chatterbox_for_fast_mps
import traceback

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


@dataclass
class GenerationConfig:
    """Configuration for audio generation parameters."""
    temperature: float = 0.8
    cfg_weight: float = 0.5
    min_p: float = 0.05
    top_p: float = 1.0
    repetition_penalty: float = 1.2
    steps: int = 1000
    exaggeration: float = 0.5


@dataclass
class ChunkingConfig:
    """Configuration for audio chunking parameters."""
    enable_chunking: bool = True
    chunk_size: int = 250


def set_seed(seed: int):
    """
    Set the random seed for PyTorch, CUDA, Python, and NumPy to ensure reproducible results across runs.
    
    Parameters:
        seed (int): The seed value to use for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def split_text_into_chunks(text: str, max_chars: int = 250) -> List[str]:
    """
    Splits input text into chunks not exceeding a specified character limit, prioritizing sentence boundaries.
    
    Handles texts of any length efficiently by processing incrementally. If a sentence exceeds the maximum 
    character limit, it is further split by commas, and then by spaces as a last resort. Returns a list 
    of non-empty text chunks suitable for sequential processing.
     
    Parameters:
        text (str): The input text to be split. No length limit.
        max_chars (int): Maximum number of characters allowed per chunk. Defaults to 250.
    
    Returns:
        List[str]: List of text chunks, each not exceeding the specified character limit.
    """
    if not text or not text.strip():
        return []
        
    if len(text) <= max_chars:
        return [text]
    
    # For very long texts, log progress
    if len(text) > 10000:
        logger.info(f"   Splitting {len(text):,} characters into chunks of {max_chars} chars...")
    
    # Split by sentences first (period, exclamation, question mark)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for i, sentence in enumerate(sentences):
        # Log progress for very long texts
        if len(sentences) > 100 and i % 100 == 0:
            logger.info(f"   Processing sentence {i}/{len(sentences)}...")
            
        # If single sentence is too long, split by commas or spaces
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence by commas
            parts = re.split(r'(?<=,)\s+', sentence)
            for part in parts:
                if len(part) > max_chars:
                    # Split by spaces as last resort
                    words = part.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk + " " + word) <= max_chars:
                            word_chunk += " " + word if word_chunk else word
                        else:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word
                    if word_chunk:
                        chunks.append(word_chunk.strip())
                else:
                    if len(current_chunk + " " + part) <= max_chars:
                        current_chunk += " " + part if current_chunk else part
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part
        else:
            # Normal sentence processing
            if len(current_chunk + " " + sentence) <= max_chars:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Filter out empty chunks and log final count
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    if len(text) > 10000:
        logger.info(f"   Split complete: {len(chunks)} chunks created")
    
    return chunks


def warmup_model(model):
    """
    Performs a sample audio generation to precompile model kernels and verify readiness.
    
    Returns:
        bool: True if the model successfully generates non-empty audio, False otherwise.
    """
    logger.info("🔥 Starting model warm-up...")
    try:
        start_time = time.time()
        # Use a realistic sentence for warm-up
        warmup_text = "Hello, this is a test to warm up the model for faster inference."
        
        # Generate audio
        wav = model.generate(
            warmup_text, 
            audio_prompt_path=None, 
            temperature=0.7, 
            cfg_weight=0.5,
            min_p=0.05,
            top_p=1.0,
            repetition_penalty=1.2
        )
        
        # Verify output
        if wav is None or wav.shape[-1] == 0:
            logger.error("❌ Warm-up failed: No audio generated")
            return False
            
        duration = wav.shape[-1] / model.sr
        end_time = time.time()
        logger.info(f"✅ Model warm-up completed in {end_time - start_time:.2f} seconds")
        logger.info(f"   Generated {duration:.2f} seconds of audio")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model warm-up failed: {e}", exc_info=True)
        return False


def transfer_model_to_gpu(model, device):
    """
    Transfers all components of a ChatterboxTTS model to the specified GPU or MPS device.
    
    Moves the model's core modules and conditionals to the target device, applies device-specific optimizations if needed, and ensures transfer completion. Raises an exception if the transfer fails.
    
    Parameters:
        device (str): Target device identifier, such as 'cuda' or 'mps'.
    
    Returns:
        model: The ChatterboxTTS model with all components moved to the specified device.
    """
    logger.info(f"📤 Transferring model components from CPU to {device}...")
    start_time = time.time()
    
    try:
        # Transfer each component
        logger.info("   - Transferring T3 model...")
        model.t3 = model.t3.to(device)
        
        logger.info("   - Transferring S3Gen model...")
        model.s3gen = model.s3gen.to(device)
        
        logger.info("   - Transferring Voice Encoder...")
        model.ve = model.ve.to(device)
        
        # Update device attribute
        model.device = device
        
        # Transfer conditionals if they exist
        if model.conds is not None:
            logger.info("   - Transferring conditionals...")
            model.conds = model.conds.to(device)
        
        # Apply MPS-specific optimizations if needed
        if device == 'mps':
            logger.info("   - Applying fast MPS optimizations...")
            model = optimize_chatterbox_for_fast_mps(model)
        
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
    Loads the ChatterboxTTS model, transfers it to the appropriate device, applies optimizations, and performs a warm-up to ensure readiness.
    
    Returns:
        model (ChatterboxTTS): The fully initialized and optimized TTS model ready for inference.
    
    Raises:
        RuntimeError: If the model fails to warm up successfully.
    """
    logger.info("=" * 60)
    logger.info("🚀 INITIALIZING CHATTERBOX TTS MODEL")
    logger.info("=" * 60)
    
    # Step 1: Load model on CPU
    logger.info("📥 Step 1: Loading model on CPU...")
    start_time = time.time()
    model = ChatterboxTTS.from_pretrained("cpu")
    logger.info(f"✅ Model loaded on CPU in {time.time() - start_time:.2f} seconds")
    
    # Step 2: Transfer to GPU if available
    if DEVICE != "cpu":
        logger.info(f"📤 Step 2: Transferring model to {DEVICE}...")
        model = transfer_model_to_gpu(model, DEVICE)
    else:
        logger.info("ℹ️  Step 2: Skipping GPU transfer (CPU-only mode)")
    
    # Step 3: Apply BetterTransformer optimization
    logger.info("⚡ Step 3: Applying BetterTransformer optimization...")
    try:
        opt_start = time.time()
        model.t3 = BetterTransformer.transform(model.t3)
        model.s3gen = BetterTransformer.transform(model.s3gen)
        logger.info(f"✅ BetterTransformer applied in {time.time() - opt_start:.2f} seconds")
    except Exception as e:
        logger.warning(f"⚠️  Could not apply BetterTransformer: {e}")
    
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


def _process_chunks(chunks: List[str], ref_wav: str, generation_config: GenerationConfig, seed_num: int) -> List[np.ndarray]:
    """Process text chunks and return audio chunks."""
    global GLOBAL_MODEL
    audio_chunks = []
    total_start_time = time.time()

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info(f"   Chunk {i+1}/{len(chunks)}: '{chunk[:50]}...' ({len(chunk)} chars)")
        
        chunk_start_time = time.time()
        
        # Set seed if provided
        if seed_num != 0:
            set_seed(int(seed_num))
        
        # Generate audio for this chunk
        result = GLOBAL_MODEL.generate(
            text=chunk,
            audio_prompt_path=ref_wav,
            exaggeration=generation_config.exaggeration,
            temperature=generation_config.temperature,
            cfg_weight=generation_config.cfg_weight,
            min_p=generation_config.min_p,
            top_p=generation_config.top_p,
            repetition_penalty=generation_config.repetition_penalty,
            steps=generation_config.steps
        )
        
        if result is None:
            logger.error(f"   ❌ Failed to generate chunk {i+1}")
            continue
        
        # The model returns just the audio tensor, not (sr, audio)
        audio = result.squeeze(0).cpu().numpy()
        audio_chunks.append(audio)
        
        # Log chunk timing
        chunk_time = time.time() - chunk_start_time
        chunk_duration = len(audio) / GLOBAL_MODEL.sr
        chunk_rtf = chunk_time / chunk_duration
        
        if len(chunks) > 1:
            logger.info(f"   ✓ Generated {chunk_duration:.2f}s in {chunk_time:.2f}s (RTF: {chunk_rtf:.2f})")
        
        # Progress update for very long texts
        if len(chunks) > 20 and (i + 1) % 10 == 0:
            elapsed = time.time() - total_start_time
            avg_per_chunk = elapsed / (i + 1)
            remaining = avg_per_chunk * (len(chunks) - i - 1)
            logger.info(f"   Progress: {i+1}/{len(chunks)} chunks complete. Est. time remaining: {remaining:.1f}s")
            
    return audio_chunks


def _combine_audio_chunks(audio_chunks: List[np.ndarray]) -> np.ndarray:
    """Combine multiple audio chunks into a single array."""
    if len(audio_chunks) > 1:
        return np.concatenate(audio_chunks)
    if audio_chunks:
        return audio_chunks[0]
    return np.array([])


def generate(text: str, ref_wav: str, generation_config: GenerationConfig, chunking_config: ChunkingConfig, seed_num: int):
    """Generate audio with improved device management and performance"""
    global GLOBAL_MODEL
    
    try:
        # Input validation
        if not text or not text.strip():
            logger.error("❌ No text provided")
            return None
            
        # Clean the text
        text = text.strip()
        
        # Log generation details
        logger.info(f"🎤 Generating audio for text: '{text[:50]}...' ({len(text):,} characters)")
        if ref_wav:
            logger.info(f"   Using reference audio: {ref_wav}")
        logger.info(f"   Parameters: temp={generation_config.temperature}, cfg={generation_config.cfg_weight}, exaggeration={generation_config.exaggeration}, max_steps={generation_config.steps}")
        
        # Split text into chunks if enabled and text is longer than chunk size
        if chunking_config.enable_chunking and len(text) > chunking_config.chunk_size:
            chunks = split_text_into_chunks(text, max_chars=chunking_config.chunk_size)
            logger.info(f"   Processing {len(chunks)} text chunk(s)")
            
            # For very long texts, warn about processing time
            if len(chunks) > 20:
                logger.info(f"   ⚠️  Large text: {len(chunks)} chunks will be processed. This may take several minutes...")
        else:
            chunks = [text]
            if len(text) > 1000:
                logger.info(f"   Processing single chunk of {len(text):,} characters (chunking disabled)")
        
        total_start_time = time.time()

        # Process chunks
        audio_chunks = _process_chunks(
            chunks=chunks,
            ref_wav=ref_wav,
            generation_config=generation_config,
            seed_num=seed_num
        )
        
        if not audio_chunks:
            logger.error("❌ No audio chunks generated")
            return None
        
        # Combine all audio chunks
        combined_audio = _combine_audio_chunks(audio_chunks)

        # Log final timing
        total_time = time.time() - total_start_time
        if len(combined_audio) > 0:
            total_duration = len(combined_audio) / GLOBAL_MODEL.sr
            total_rtf = total_time / total_duration
            logger.info(f"✅ Total: Generated {total_duration:.2f}s of audio in {total_time:.2f}s (RTF: {total_rtf:.2f})")
        else:
             logger.info(f"✅ Total: Generation finished in {total_time:.2f}s, but no audio was produced.")

        return GLOBAL_MODEL.sr, combined_audio
        
    except Exception as e:
        logger.error(f"❌ Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        return None


# Initialize model before creating the interface
logger.info("🚀 Starting Chatterbox TTS Application...")
try:
    GLOBAL_MODEL = load_and_prepare_model()
except Exception as e:
    logger.error(f"Failed to initialize model: {e}", exc_info=True)
    raise

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎤 Chatterbox TTS")
    gr.Markdown("Model loaded and warmed up. Ready for fast inference!")
    
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (no character limit)",
                max_lines=None,
                lines=10,
                placeholder="Enter any amount of text here..."
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5)", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)
            steps = gr.Slider(100, 2000, step=50, label="Generation Steps", value=1000)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)
                
                gr.Markdown("### Text Processing")
                enable_chunking = gr.Checkbox(label="Enable text chunking", value=True, info="Split long texts into smaller chunks for better quality")
                chunk_size = gr.Slider(100, 10000, step=50, label="Chunk size (characters)", value=250, visible=True)
                
                # Toggle chunk size visibility based on checkbox
                enable_chunking.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[enable_chunking],
                    outputs=[chunk_size]
                )

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            gr.Markdown("### 🎵 Generated Audio")
            gr.Markdown("*Click the download button (⬇️) in the audio player to save the generated audio file*")
            audio_output = gr.Audio(
                label="Output Audio",
                type="numpy",
                show_download_button=True,
                format="wav"
            )
            
            # Add generation info display
            generation_info = gr.Markdown(visible=False)

    def generate_with_info(text, ref_wav, exaggeration, cfg_weight, steps, seed_num, temp, min_p, top_p, repetition_penalty, enable_chunking, chunk_size):
        """Wrapper to add generation info display"""
        import time
        # Extract text from args to check length
        text_length = len(text)
        
        start_time = time.time()

        generation_config = GenerationConfig(
            temperature=temp,
            cfg_weight=cfg_weight,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            steps=steps,
            exaggeration=exaggeration
        )
        chunking_config = ChunkingConfig(
            enable_chunking=enable_chunking,
            chunk_size=chunk_size
        )

        result = generate(
            text=text,
            ref_wav=ref_wav,
            generation_config=generation_config,
            chunking_config=chunking_config,
            seed_num=seed_num
        )
        end_time = time.time()
        
        if result is not None and result[1].size > 0:
            sr, audio = result
            duration = len(audio) / sr
            generation_time = end_time - start_time
            rtf = generation_time / duration if duration > 0 else 0
            
            info_text = f"""✅ **Generation complete!**
- Input text: {text_length:,} characters
- Audio duration: {duration:.2f} seconds
- Generation time: {generation_time:.2f} seconds
- Real-time factor (RTF): {rtf:.2f}x
- Sample rate: {sr:,} Hz
- Click the download button above to save"""
            
            return result, gr.update(value=info_text, visible=True)
        else:
            return (24000, np.array([])), gr.update(value="❌ Generation failed", visible=True)

    run_btn.click(
        fn=generate_with_info,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            cfg_weight,
            steps,
            seed_num,
            temp,
            min_p,
            top_p,
            repetition_penalty,
            enable_chunking,
            chunk_size,
        ],
        outputs=[audio_output, generation_info],
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=False)
