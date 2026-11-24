# Orukeet transcription API

Source and server installations can opt into Orukeet, a local 25-language speech recognizer based on Parakeet TDT v3. Whisper remains the default. This addition exposes Orukeet through the transcription API; the desktop model picker and bundled installers still use Whisper.

Install the optional CPU runtime in the same environment as the Voicebox server:

```bash
pip install -r backend/requirements-orukeet.txt
```

Select it for a recording:

```bash
curl http://localhost:17493/transcribe \
  -F 'file=@recording.wav' -F 'model=orukeet'
```

The response retains the existing `text` and `duration` fields. The first request downloads about 640 MiB of INT8 ONNX files from the pinned [Hugging Face release](https://huggingface.co/oruk/orukeet/tree/1751fce6ecde442f14543cf1804800c49b3e415c/onnx/combined-v0.1.0-int8) and verifies their SHA-256 hashes. Allow extra time for that first request. Later requests reuse the model; restarting the server reuses the Hugging Face cache without contacting the network when all required files are present.

The required `config.json` download participates in Hugging Face's normal model download accounting. Voicebox sends no audio to Hugging Face or an Oruk service. The model remains loaded in CPU memory until the server exits.

Orukeet detects the spoken language automatically. The optional `language` field validates that the language is supported; it does not force decoding in that language. Supported codes: `bg`, `hr`, `cs`, `da`, `nl`, `en`, `et`, `fi`, `fr`, `de`, `el`, `hu`, `it`, `lv`, `lt`, `mt`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `es`, `sv`, `uk`. This endpoint transcribes complete recordings and does not provide streaming, translation or word timestamps.

Weights use CC BY-SA 4.0. The download includes the weight license and notices, converter MIT license and preprocessor attribution.
