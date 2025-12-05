import time
import uuid
import copy
import requests
from typing import Dict, Any


class Tools:

    COMFY_HOST = "http://127.0.0.1:8188"

    # ------------------------------------------------------------
    # EMBEDDED WAN 2.2 API WORKFLOW (FROM YOUR EXPORT)
    # ------------------------------------------------------------
    BASE_WORKFLOW = {
        "71": {
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
            "class_type": "CLIPLoader",
        },
        "72": {
            "inputs": {"text": "", "clip": ["71", 0]},
            "class_type": "CLIPTextEncode",
        },
        "73": {
            "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
            "class_type": "VAELoader",
        },
        "74": {
            "inputs": {"width": 640, "height": 640, "length": 81, "batch_size": 1},
            "class_type": "EmptyHunyuanLatentVideo",
        },
        "75": {
            "inputs": {
                "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "76": {
            "inputs": {
                "unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "78": {
            "inputs": {
                "add_noise": "disable",
                "noise_seed": 0,
                "steps": 4,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 2,
                "end_at_step": 4,
                "return_with_leftover_noise": "disable",
                "model": ["86", 0],
                "positive": ["89", 0],
                "negative": ["72", 0],
                "latent_image": ["81", 0],
            },
            "class_type": "KSamplerAdvanced",
        },
        "80": {
            "inputs": {
                "filename_prefix": "video/ComfyUI",
                "format": "auto",
                "codec": "auto",
                "video": ["88", 0],
            },
            "class_type": "SaveVideo",
        },
        "81": {
            "inputs": {
                "add_noise": "enable",
                "noise_seed": 0,
                "steps": 4,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 0,
                "end_at_step": 2,
                "return_with_leftover_noise": "enable",
                "model": ["82", 0],
                "positive": ["89", 0],
                "negative": ["72", 0],
                "latent_image": ["74", 0],
            },
            "class_type": "KSamplerAdvanced",
        },
        "82": {
            "inputs": {"shift": 5.000000000000001, "model": ["83", 0]},
            "class_type": "ModelSamplingSD3",
        },
        "83": {
            "inputs": {
                "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
                "strength_model": 1.0,
                "model": ["75", 0],
            },
            "class_type": "LoraLoaderModelOnly",
        },
        "85": {
            "inputs": {
                "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
                "strength_model": 1.0,
                "model": ["76", 0],
            },
            "class_type": "LoraLoaderModelOnly",
        },
        "86": {
            "inputs": {"shift": 5.000000000000001, "model": ["85", 0]},
            "class_type": "ModelSamplingSD3",
        },
        "87": {
            "inputs": {"samples": ["78", 0], "vae": ["73", 0]},
            "class_type": "VAEDecode",
        },
        "88": {"inputs": {"fps": 16, "images": ["87", 0]}, "class_type": "CreateVideo"},
        "89": {
            "inputs": {"text": "", "clip": ["71", 0]},
            "class_type": "CLIPTextEncode",
        },
    }

    DEFAULT_NEG = (
        "low quality, blurry, distorted, bad anatomy, watermark, "
        "jpeg artifacts, motion jitters"
    )

    # ------------------------------------------------------------
    # MAIN TOOL ENTRYPOINT
    # ------------------------------------------------------------
    def comfy_generate_advanced(self, prompt: str) -> Dict[str, Any]:
        try:
            workflow = copy.deepcopy(self.BASE_WORKFLOW)

            parsed = self._interpret_prompt(prompt)
            self._apply_overrides(workflow, parsed)

            prompt_id = self._send_prompt(workflow)
            video_url = self._poll_for_output(prompt_id)

            return {
                "status": "success",
                "prompt_id": prompt_id,
                "video_url": video_url,
                "duration_frames": workflow["74"]["inputs"]["length"],
                "fps": workflow["88"]["inputs"]["fps"],
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------
    # PROMPT PARSING
    # ------------------------------------------------------------
    def _interpret_prompt(self, prompt: str) -> Dict[str, Any]:
        p = prompt.lower()

        # Duration override in seconds (2–20s)
        duration = 10
        for tok in p.split():
            if tok.isdigit():
                n = int(tok)
                if 2 <= n <= 20:
                    duration = n
                    break

        # Negative override
        negative = self.DEFAULT_NEG
        if "negative:" in p:
            try:
                negative = prompt.split("negative:", 1)[1].strip()
            except:
                pass

        return {
            "positive": prompt,
            "negative": negative,
            "seconds": duration,
            "seed": int(time.time() * 1000) % 1000000000,
        }

    # ------------------------------------------------------------
    # WORKFLOW OVERRIDES
    # ------------------------------------------------------------
    def _apply_overrides(self, wf: Dict[str, Any], p: Dict[str, Any]):

        # Positive prompt
        wf["89"]["inputs"]["text"] = p["positive"]

        # Negative prompt
        wf["72"]["inputs"]["text"] = p["negative"]

        # Duration in frames (Wan uses latent length, not seconds)
        # 16 fps × seconds
        wf["74"]["inputs"]["length"] = p["seconds"] * 16

        # Seed propagation
        wf["81"]["inputs"]["noise_seed"] = p["seed"]
        wf["78"]["inputs"]["noise_seed"] = p["seed"]

        # Unique output filename
        wf["80"]["inputs"]["filename_prefix"] = f"video/owui_{p['seed']}"

    # ------------------------------------------------------------
    # COMFY API
    # ------------------------------------------------------------
    def _send_prompt(self, workflow: Dict[str, Any]) -> str:
        prompt_id = str(uuid.uuid4())

        payload = {"prompt": workflow, "client_id": prompt_id}

        r = requests.post(f"{self.COMFY_HOST}/prompt", json=payload, timeout=1200)

        if r.status_code != 200:
            raise Exception(f"ComfyUI error {r.status_code}: {r.text}")

        return prompt_id

    def _poll_for_output(self, prompt_id: str) -> str:
        history_url = f"{self.COMFY_HOST}/history/{prompt_id}"

        for _ in range(600):  # ~5 minutes
            time.sleep(0.5)
            r = requests.get(history_url)
            if r.status_code != 200:
                continue

            data = r.json()
            if "outputs" not in data:
                continue

            for o in data["outputs"].values():
                if "files" in o and o["files"]:
                    fname = o["files"][-1]["name"]
                    return f"{self.COMFY_HOST}/view?filename={fname}"

        raise Exception("Timed out waiting for ComfyUI video output.")
