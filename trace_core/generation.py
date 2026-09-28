import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import re
from typing import Callable, Tuple, Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = None, revision: str = None, num_threads: int = 8):
        self.model_name = model_name
        self.revision = revision

        # Set PyTorch CPU inference threads
        if num_threads > 0 and not torch.cuda.is_available():
            torch.set_num_threads(num_threads)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.float32 if not torch.cuda.is_available() else "auto",
            device_map="auto" if device is None and torch.cuda.is_available() else None,
        )
        if device is not None:
            self.model.to(device)
        elif not torch.cuda.is_available():
            self.model.to("cpu")

        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 256, seed: int = 13, top_p: float = 0.95):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        start = time.perf_counter()

        is_greedy = (temperature <= 1e-4)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not is_greedy,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if not is_greedy:
            kwargs["temperature"] = max(temperature, 0.01)
            kwargs["top_p"] = top_p

        with torch.no_grad():
            gen_out = self.model.generate(**inputs, **kwargs)

        elapsed = time.perf_counter() - start
        sequences = gen_out.sequences
        generated = sequences[0, input_tokens:]
        decoded_text = self.tokenizer.decode(generated, skip_special_tokens=True)

        confidence = 1.0
        if gen_out.scores:
            import torch.nn.functional as F
            probs = []
            for s, t in zip(gen_out.scores, generated):
                p = F.softmax(s, dim=-1)[0, t].item()
                probs.append(p)
            if probs:
                confidence = sum(probs) / len(probs)

        return {
            "text": decoded_text,
            "confidence": round(confidence, 4),
            "input_tokens": input_tokens,
            "output_tokens": int(generated.shape[-1]),
            "latency_seconds": elapsed,
        }

    def stream_generate(
        self,
        prompt: str,
        is_target_complete_fn: Callable[[str], Tuple[bool, Optional[str]]],
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        seed: int = 13,
        top_p: float = 0.95
    ) -> Dict[str, Any]:
        """Stream/generate tokens and halt early using StoppingCriteria when target is emitted."""
        from transformers import StoppingCriteria, StoppingCriteriaList

        class EarlyStoppingCriteria(StoppingCriteria):
            def __init__(self, tokenizer, check_fn, p_len):
                super().__init__()
                self.tokenizer = tokenizer
                self.check_fn = check_fn
                self.p_len = p_len
                self.action = None

            def __call__(self, input_ids, scores, **kwargs):
                gen = input_ids[0, self.p_len:]
                txt = self.tokenizer.decode(gen, skip_special_tokens=True)
                done, act = self.check_fn(txt)
                if done:
                    self.action = act
                    return True
                return False

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        start = time.perf_counter()

        is_greedy = (temperature <= 1e-4)
        early_stopper = EarlyStoppingCriteria(self.tokenizer, is_target_complete_fn, input_tokens)

        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not is_greedy,
            "pad_token_id": self.tokenizer.pad_token_id,
            "stopping_criteria": StoppingCriteriaList([early_stopper]),
        }
        if not is_greedy:
            kwargs["temperature"] = max(temperature, 0.01)
            kwargs["top_p"] = top_p

        with torch.no_grad():
            gen_out = self.model.generate(**inputs, **kwargs)

        elapsed = time.perf_counter() - start
        generated = gen_out[0, input_tokens:]
        decoded_text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return {
            "text": decoded_text,
            "action": early_stopper.action,
            "tokens": int(generated.shape[-1]),
            "early_halt": early_stopper.action is not None,
            "latency_seconds": elapsed,
        }



class GroqGenerator:
    """High-speed cloud inference generator via Groq API."""
    def __init__(self, model_name: str = "qwen/qwen3.8-27b", api_key: str = None):
        from groq import Groq
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
        if not api_key and os.path.exists(".env"):
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GROQ_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip()
                        break
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please provide it or set it in .env.")
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key, timeout=25.0)
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 256, seed: int = 13, top_p: float = 0.95):
        from groq import RateLimitError, APIError, APIConnectionError
        is_greedy = (temperature <= 1e-4)
        temp = 1e-5 if is_greedy else max(temperature, 0.01)
        start = time.perf_counter()

        max_retries = 8
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=temp,
                    max_tokens=max_new_tokens,
                    seed=seed,
                    top_p=1.0 if is_greedy else top_p,
                )
                elapsed = time.perf_counter() - start
                choice = response.choices[0]
                text = choice.message.content or ""
                input_toks = response.usage.prompt_tokens if response.usage else 0
                output_toks = response.usage.completion_tokens if response.usage else 0

                confidence = 1.0 if is_greedy else 0.88
                # Penalize confidence if model exhibits verbal uncertainty markers
                lower_text = text.lower()
                if any(m in lower_text for m in ["maybe", "not sure", "possibly", "could be", "uncertain"]):
                    confidence = 0.45

                return {
                    "text": text,
                    "confidence": confidence,
                    "input_tokens": input_toks,
                    "output_tokens": output_toks,
                    "latency_seconds": elapsed,
                }
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                err_msg = str(e)
                wait_time = backoff
                match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", err_msg)
                if match:
                    mins = float(match.group(1) or 0)
                    secs = float(match.group(2) or 0)
                    wait_time = mins * 60 + secs + 2.0
                time.sleep(wait_time)
                backoff = min(backoff * 1.5, 15.0)
            except (APIConnectionError, APIError, Exception) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 15.0)

    def stream_generate(
        self,
        prompt: str,
        is_target_complete_fn: Callable[[str], Tuple[bool, Optional[str]]],
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        seed: int = 13,
        top_p: float = 0.95
    ) -> Dict[str, Any]:
        """Stream tokens and halt immediately as soon as is_target_complete_fn detects the target."""
        from groq import RateLimitError, APIError, APIConnectionError
        is_greedy = (temperature <= 1e-4)
        temp = 1e-5 if is_greedy else max(temperature, 0.01)
        start = time.perf_counter()

        max_retries = 8
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                stream = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=temp,
                    max_tokens=max_new_tokens,
                    seed=seed,
                    top_p=1.0 if is_greedy else top_p,
                    stream=True,
                )
                accumulated_text = ""
                tokens_emitted = 0
                early_action = None

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content or ""
                    accumulated_text += delta
                    tokens_emitted += 1
                    is_complete, extracted = is_target_complete_fn(accumulated_text)
                    if is_complete:
                        early_action = extracted
                        try:
                            stream.close()
                        except Exception:
                            pass
                        break

                elapsed = time.perf_counter() - start
                time.sleep(0.2)

                return {
                    "text": accumulated_text,
                    "action": early_action,
                    "tokens": tokens_emitted,
                    "early_halt": early_action is not None,
                    "latency_seconds": elapsed,
                }
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                err_msg = str(e)
                wait_time = backoff
                match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", err_msg)
                if match:
                    mins = float(match.group(1) or 0)
                    secs = float(match.group(2) or 0)
                    wait_time = mins * 60 + secs + 2.0
                time.sleep(wait_time)
                backoff = min(backoff * 1.5, 15.0)
            except (APIConnectionError, APIError, Exception) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 15.0)



