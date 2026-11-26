"""
Before vs After demo: original Mistral-7B-Instruct v0.3 vs. the fine-tuned
customer-support model, answering the same question side by side.

Both models run locally through Ollama (https://ollama.com):
  - ORIGINAL_MODEL: the untrained base model   (default: "mistral-original")
  - TUNED_MODEL:    the fine-tuned GGUF model   (default: "support-agent")

Run:  python app.py            -> opens http://127.0.0.1:7860
      SHARE=1 python app.py    -> also creates a temporary public link
"""

import json
import os
import re
import time

import gradio as gr
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ORIGINAL_MODEL = os.getenv("ORIGINAL_MODEL", "mistral-original")
TUNED_MODEL = os.getenv("TUNED_MODEL", "support-agent")

# Same rules as the evaluation in the notebooks: no randomness, max 400 new tokens.
GEN_OPTIONS = {"temperature": 0, "num_predict": 400, "seed": 42}

PLACEHOLDER = re.compile(r"\{\{[^{}]+\}\}")

EXAMPLES = [
    "I want to cancel my order, how do I do it?",
    "get invoice #00108",
    "need to talk to customer support",
    "I was charged twice for the same order",
    "how can I change my shipping address?",
    "i forgot my password, help",
]

HEADER = f"""
# Mistral-7B: before vs after fine-tuning
Type a customer message (or click an example). The **same question** goes to:

- **A. Original Mistral-7B-Instruct v0.3**: the base model, not trained by me
- **C. My fine-tuned model**: the same base model + a QLoRA adapter trained on 4,050 Bitext customer-support conversations

Both are the same size and compression (GGUF Q4_K_M) and use the same settings (temperature 0, max 400 tokens),
so **the only difference is the fine-tuning**. `{{{{placeholders}}}}` are blanks the company's system would fill in
(order numbers, phone numbers, etc.) instead of the model inventing details.
"""

FOOTER = """
---
Models run locally with [Ollama](https://ollama.com).
Fine-tuned adapter: [pramod586/mistral-7b-customer-support-lora](https://huggingface.co/pramod586/mistral-7b-customer-support-lora) ·
GGUF: [pramod586/mistral-7b-customer-support-GGUF](https://huggingface.co/pramod586/mistral-7b-customer-support-GGUF)
"""


def check_models():
    """Fail early with a clear message if Ollama or a model is missing."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException:
        raise SystemExit(
            f"Cannot reach Ollama at {OLLAMA_URL}. "
            "Open the Ollama app (or run `ollama serve`) and try again."
        )
    installed = {m["name"] for m in r.json().get("models", [])}
    for model in (ORIGINAL_MODEL, TUNED_MODEL):
        if model not in installed and f"{model}:latest" not in installed:
            raise SystemExit(
                f"Model '{model}' is not in Ollama. Installed: {sorted(installed)}"
            )
    print(f"OK: Ollama is running with '{ORIGINAL_MODEL}' and '{TUNED_MODEL}'.")


def stream_answer(model, question, keep_alive="5m"):
    """Ask one model through Ollama and yield the answer as it grows."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": True,
        "options": GEN_OPTIONS,
        "keep_alive": keep_alive,
    }
    text = ""
    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            text += chunk.get("message", {}).get("content", "")
            yield text


def highlight(text):
    """Show {{placeholders}} as inline code so they stand out."""
    return PLACEHOLDER.sub(lambda m: f"`{m.group(0)}`", text)


def stats(text, seconds):
    return (
        f"⏱ {seconds:.1f}s · {len(text.split())} words · "
        f"{len(PLACEHOLDER.findall(text))} placeholders"
    )


def compare(question):
    question = (question or "").strip()
    if not question:
        yield "Please type a customer message.", "", "", ""
        return

    waiting = "_Waiting for the original model to finish..._"
    original, tuned = "", ""
    try:
        # 1) Original model first. keep_alive=0 unloads it right after,
        #    freeing memory for the fine-tuned model (helps on 8 GB machines).
        start = time.time()
        for original in stream_answer(ORIGINAL_MODEL, question, keep_alive=0):
            yield highlight(original), "⏳ generating...", waiting, ""
        original_stats = stats(original, time.time() - start)

        # 2) Fine-tuned model.
        start = time.time()
        for tuned in stream_answer(TUNED_MODEL, question):
            yield highlight(original), original_stats, highlight(tuned), "⏳ generating..."
        yield highlight(original), original_stats, highlight(tuned), stats(tuned, time.time() - start)
    except (requests.RequestException, RuntimeError) as e:
        raise gr.Error(f"Ollama error: {e}")


with gr.Blocks(title="Mistral-7B: before vs after fine-tuning") as demo:
    gr.Markdown(HEADER)
    question = gr.Textbox(
        label="Customer message",
        placeholder="e.g. I want to cancel my order, how do I do it?",
        lines=2,
    )
    button = gr.Button("Compare answers", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### A. Original Mistral-7B (before)")
            original_out = gr.Markdown()
            original_stats = gr.Markdown()
        with gr.Column():
            gr.Markdown("### C. Fine-tuned support agent (after)")
            tuned_out = gr.Markdown()
            tuned_stats = gr.Markdown()
    gr.Examples(examples=[[q] for q in EXAMPLES], inputs=question)
    gr.Markdown(FOOTER)

    outputs = [original_out, original_stats, tuned_out, tuned_stats]
    button.click(compare, inputs=question, outputs=outputs)
    question.submit(compare, inputs=question, outputs=outputs)


if __name__ == "__main__":
    check_models()
    demo.queue().launch(share=os.getenv("SHARE") == "1")
