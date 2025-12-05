import time
import json
import requests

class Tools:
    """
    Generate a text-to-video clip using ComfyUI (Wan 2.2 default workflow).
    Automatically waits for completion (up to 15 minutes) and returns a video URL.
    """

    COMFY_HOST = "http://127.0.0.1:8188"
    POLL_INTERVAL = 5       # seconds
    MAX_WAIT = 15 * 60      # 15 minutes

    def run(self, prompt: str, negative_prompt: str = None, duration_seconds: int = 10):
        """
        Generate a video from text using a static Wan 2.2 workflow.
        """

        if not negative_prompt:
            negative_prompt = "blurry, low quality, jpeg artifacts, distorted, ugly"

        # --- STATIC WAN 2.2 WORKFLOW TEMPLATE (MINIMAL) ---
        workflow = {
            "71": {
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "type": "wan",
                    "device": "default"
                },
                "class_type": "CLIPLoader"
            },
            "72": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["71", 0]
                },
                "class_type": "CLIPTextEncode"
            },
            "73": {
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors"
                },
                "class_type": "VAELoader"
            },
            "74": {
                "inputs": {
                    "width": 640,
                    "height": 640,
                    "length": int(duration_seconds * 8),  # ~8 fps latent length
                    "batch_size": 1
                },
                "class_type": "EmptyHunyuanLatentVideo"
            },
            "75": {
                "inputs": {
                    "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader"
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
                    "latent_image": ["74", 0]
                },
                "class_type": "KSamplerAdvanced"
            },
            "87": {
                "inputs": {
                    "samples": ["78", 0],
                    "vae": ["73", 0]
                },
                "class_type": "VAEDecode"
            },
            "88": {
                "inputs": {
                    "fps": 16,
                    "images": ["87", 0]
                },
                "class_type": "CreateVideo"
            },
            "80": {
                "inputs": {
                    "filename_prefix": "video/wan2_2",
                    "format": "auto",
                    "codec": "auto",
                    "video": ["88", 0]
                },
                "class_type": "SaveVideo"
            },
            "89": {
                "inputs": {
                    "text": prompt,
                    "clip": ["71", 0]
                },
                "class_type": "CLIPTextEncode"
            }
        }

        payload = {
            "prompt": workflow
        }

        # --- SUBMIT WORKFLOW ---
        r = requests.post(
            f"{self.COMFY_HOST}/prompt",
            json=payload,
            timeout=30
        )

        if r.status_code != 200:
            raise Exception(f"ComfyUI submit failed: {r.status_code} {r.text}")

        resp_json = r.json()
        prompt_id = resp_json.get("prompt_id")

        if not prompt_id:
            raise Exception(f"No prompt_id returned by ComfyUI: {resp_json}")

        # --- POLL FOR OUTPUT (MANUAL) ---
        deadline = time.time() + self.MAX_WAIT

        while time.time() < deadline:
            time.sleep(self.POLL_INTERVAL)

            hr = requests.get(
                f"{self.COMFY_HOST}/history/{prompt_id}",
                timeout=30
            )

            if hr.status_code != 200:
                continue

            history = hr.json()
            outputs = history.get("outputs", {})

            for node in outputs.values():
                files = node.get("files", [])
                if files:
                    fname = files[-1].get("name")
                    if fname:
                        video_url = f"{self.COMFY_HOST}/view?filename={fname}"
                        return {
                            "status": "completed",
                            "prompt_id": prompt_id,
                            "video_url": video_url
                        }

        # --- TIMEOUT FALLBACK ---
        return {
            "status": "processing",
            "prompt_id": prompt_id,
            "message": "Video is still rendering. Call this tool again later with the same prompt."
        }

