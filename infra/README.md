# infra/ – Phases 8–10 (deferred)

Folders for later phases. Docker is not needed until Phase 8, and the decision to install it is made then.

- `terraform/`: Docker provider stack plus one GCS bucket, parameterised module
- `mediamtx/`, `frigate/`: simulated RTSP feed and recorder (fallback: MediaMTX + ffmpeg segments)
- `triton/model_repo/`: student exported to ONNX, CPU backend
