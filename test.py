import soundfile as sf
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")

text = "VoxCPM is an innovative end-to-end TTS model from ModelBest."

for i in range(3):
    wav = model.generate(
        text=text,
        prompt_wav_path=None,   # ← 不提供参考音频
        prompt_text=None,       # ← 也不提供参考文本
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=True,
        denoise=True,
    )
    sf.write(f"no_prompt_run_{i+1}.wav", wav, 16000)
    print(f"Saved: no_prompt_run_{i+1}.wav")