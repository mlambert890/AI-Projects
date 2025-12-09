import requests
import json
import time


class Tools:
    """
    Community-ready OWUI Tool for Text-to-Video using ComfyUI (WAN 2.2)
    - Submits a workflow
    - Returns the real ComFY prompt_id
    - Polls true completion status
    - Correctly detects completed videos from history
    - Prevents OWUI from rewriting Comfy URLs
    """

    COMFY_HOST = "http://127.0.0.1:8188"
    POLL_INTERVAL = 5
    MAX_WAIT = 15 * 60

    def run(self, prompt: str, negative_prompt: str = None, duration_seconds: int = 10):
        prompt = prompt.strip()

        if prompt.lower().startswith("check "):
            prompt_id = prompt[6:].strip()
            return self._check_status(prompt_id)

        return self._submit_workflow(prompt, negative_prompt, duration_seconds)

    # -----------------------------
    # WORKFLOW SUBMISSION
    # -----------------------------
    def _submit_workflow(self, positive_prompt, negative_prompt, duration_seconds):
        if not negative_prompt:
            negative_prompt = "blurry, low quality, jpeg artifacts, distorted, ugly"

        workflow = {
            "71": {
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "type": "wan",
                    "device": "default",
                },
                "class_type": "CLIPLoader",
            },
            "72": {
                "inputs": {"text": negative_prompt, "clip": ["71", 0]},
                "class_type": "CLIPTextEncode",
            },
            "73": {
                "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
                "class_type": "VAELoader",
            },
            "74": {
                "inputs": {
                    "width": 640,
                    "height": 640,
                    "length": int(duration_seconds * 8),
                    "batch_size": 1,
                },
                "class_type": "EmptyHunyuanLatentVideo",
            },
            "75": {
                "inputs": {
                    "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                    "weight_dtype": "default",
                },
                "class_type": "UNETLoader",
            },
            "78": {
                "inputs": {
                    "add_noise": "enable",
                    "noise_seed": 0,
                    "steps": 4,
                    "cfg": 1,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": 0,
                    "end_at_step": 4,
                    "return_with_leftover_noise": "disable",
                    "model": ["75", 0],
                    "positive": ["89", 0],
                    "negative": ["72", 0],
                    "latent_image": ["74", 0],
                },
                "class_type": "KSamplerAdvanced",
            },
            "87": {
                "inputs": {"samples": ["78", 0], "vae": ["73", 0]},
                "class_type": "VAEDecode",
            },
            "88": {
                "inputs": {"fps": 16, "images": ["87", 0]},
                "class_type": "CreateVideo",
            },
            "80": {
                "inputs": {
                    "filename_prefix": "video/wan2_2",
                    "format": "auto",
                    "codec": "auto",
                    "video": ["88", 0],
                },
                "class_type": "SaveVideo",
            },
            "89": {
                "inputs": {"text": positive_prompt, "clip": ["71", 0]},
                "class_type": "CLIPTextEncode",
            },
        }

        payload = {"prompt": workflow}

        try:
            r = requests.post(f"{self.COMFY_HOST}/prompt", json=payload, timeout=30)
            r.raise_for_status()
        except Exception as e:
            return {"status": "error", "message": str(e)}

        data = r.json()
        prompt_id = data.get("prompt_id")

        print(f"[LOG] Submission Response: {json.dumps(data)}")

        if not prompt_id:
            return {"status": "error", "message": "No prompt_id returned from ComfyUI"}

        return {
            "status": "submitted",
            "prompt_id": prompt_id,
            "message": f"Submitted successfully. Use `check {prompt_id}`.",
        }

    # -----------------------------
    # ✅ STATUS CHECKER
    # -----------------------------
    def _check_status(self, prompt_id: str):
        try:
            r = requests.get(f"{self.COMFY_HOST}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
        except Exception as e:
            return {
                "status": "error",
                "prompt_id": prompt_id,
                "message": f"Failed to query ComfyUI history: {e}",
            }

        raw = r.json()
        print(f"[LOG] Full history payload: {json.dumps(raw)}")

        # ✅ History is wrapped in the prompt_id key
        entry = raw.get(prompt_id)
        if not entry:
            return {
                "status": "processing",
                "prompt_id": prompt_id,
                "message": "History entry not yet registered for this job.",
            }

        outputs = entry.get("outputs", {})
        status_block = entry.get("status", {})
        completed = status_block.get("completed", False)

        for node_id, node in outputs.items():
            files = node.get("videos") or node.get("gifs") or node.get("images") or []

            if files:
                f = files[-1]
                fname = f.get("filename")
                sub = f.get("subfolder", "")

                video_url = f"{self.COMFY_HOST}/view?filename={fname}&subfolder={sub}"

                # ✅ CRITICAL OWUI-SAFE RETURN (NO FILE-TRIGGERING KEYS)
                return {
                    "status": "completed",
                    "prompt_id": prompt_id,
                    "DOWNLOAD": video_url,
                }

        if completed:
            return {
                "status": "completed",
                "prompt_id": prompt_id,
                "message": "Job completed but no video file was attached.",
            }

        return {
            "status": "processing",
            "prompt_id": prompt_id,
            "message": "Job still running.",
        }
