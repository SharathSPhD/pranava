# Śabda-ALM — deployment

The app has two halves: a **static frontend** (public, on Vercel) and a **GPU inference
server** (runs where the model lives — the GB10 or the RTX 5090).

## Frontend (live)
- **https://sabda-alm.vercel.app** — mic recording + file upload, calls a configurable
  `/transcribe` endpoint, shows the transcription with its detected language.
- Source: `web/index.html` (+ `web/vercel.json`). Redeploy with the Vercel MCP `deploy_to_vercel`
  or `vercel --prod` from `web/`.

## Inference server (run on the GPU box)
The model needs mamba/NeMo CUDA kernels, so the server runs inside `prabhasa/nemo-gb10:26.02`:

```bash
docker run -d --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --name pranava-serve -p 8088:8088 \
  -v /home/sharaths/projects:/work -w /work \
  -e PYTHONPATH=/work/pranava/src:/work/prabhasa-samskrutam/src \
  -e HF_HOME=/work/pranava/.hf_cache -e PRANAVA_CORPUS=speech_corpus_multi \
  prabhasa/nemo-gb10:26.02 uvicorn pranava.serve.server:app --host 0.0.0.0 --port 8088
```

Verify: `curl -s --data-binary @clip.wav http://localhost:8088/transcribe`.

## Connecting the two
The GPU server is on a LAN, not the public internet. To let the Vercel page reach it, expose
`localhost:8088` with a tunnel (e.g. `cloudflared tunnel --url http://localhost:8088`) and paste
the tunnel URL + `/transcribe` into the site's **API endpoint** field. The endpoint is stored in
the browser's localStorage, so it persists per visitor. Without a tunnel the frontend is a live
UI with no reachable backend — that is the honest state: the public page is up; transcription
requires a running server the operator points it at.
