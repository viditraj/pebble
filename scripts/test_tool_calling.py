"""
Test Pebble's tool-calling ability after SFT.

Uses a two-step generation approach:
  Step 1: Check first-token probabilities after <|assistant|>
          If P('{') > threshold -> tool call path
          Otherwise -> direct response path
  Step 2a (tool call): Generate JSON, parse it, execute tool, feed result back
  Step 2b (direct): Generate natural language response with tool tokens suppressed

Usage:
    python scripts/test_tool_calling.py
"""

import os
import sys
import json
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from model.pebble import Pebble, PebbleConfig
from tokenizer.bpe import BPETokenizer
from inference.generate import generate
from data.tool_format import (
    format_tool_schema, BOS, EOS, END, SYSTEM, USER, ASSISTANT,
    TOOL_CALL_START, TOOL_CALL_END, TOOL_RESULT_START, TOOL_RESULT_END,
)


def load_model(checkpoint_path, config_path):
    """Load the SFT fine-tuned model."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    model_config = PebbleConfig(
        vocab_size=mc["vocab_size"],
        d_model=mc["d_model"],
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        n_kv_heads=mc["n_kv_heads"],
        d_ff=mc["d_ff"],
        max_seq_len=mc["max_seq_len"],
    )
    model = Pebble(model_config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    step = ckpt.get("step", "?")
    print(f"Loaded SFT checkpoint (step {step}) on {device}")
    return model


def should_call_tool(model, tokenizer, prompt_tokens, threshold=0.4):
    """
    Check if the model wants to call a tool.

    Looks at the probability distribution for the first token after <|assistant|>.
    If P('{') > threshold, the model is trying to produce a tool call JSON.
    """
    device = next(model.parameters()).device
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x)[0, -1, :]

    probs = F.softmax(logits, dim=-1)
    brace_prob = probs[123].item()

    return brace_prob > threshold, brace_prob


def parse_tool_call_json(text):
    """Extract tool call from generated text."""
    for marker in ["<|tool_call_start|>", "<|tool_call_end|>", "<|end|>", "<|eos|>"]:
        text = text.replace(marker, "")
    text = text.strip()

    if text and not text.startswith("{"):
        text = "{" + text

    try:
        call = json.loads(text)
        if "name" in call:
            return call
    except json.JSONDecodeError:
        pass
    return None


def get_suppress_tokens(tokenizer):
    """
    Token IDs to suppress when generating natural language.

    When the router decides "no tool call", we suppress JSON/tool tokens
    so the model is forced to produce natural language instead.
    This is called "logit masking" — a standard production technique.
    """
    sp = tokenizer.special_tokens
    suppress = [123, 125]  # '{' and '}'
    for name in ["<|tool_call_start|>", "<|tool_call_end|>",
                 "<|tool_result_start|>", "<|tool_result_end|>",
                 "<|system|>", "<|user|>"]:
        if name in sp:
            suppress.append(sp[name])
    return suppress


def run_tool_call(model, tokenizer, prompt_tokens, tool_functions):
    """Generate a tool call, execute it, feed result back, generate response."""
    sp = tokenizer.special_tokens
    end_id = sp["<|end|>"]

    # Generate the tool call JSON
    output_tokens = generate(
        model, prompt_tokens,
        max_new_tokens=150,
        temperature=0.0,
        eos_token_id=end_id,
    )

    gen_text = tokenizer.decode(output_tokens[len(prompt_tokens):])
    tool_call = parse_tool_call_json(gen_text)

    if not tool_call:
        return {"tool_called": None, "response": gen_text, "error": "Failed to parse"}

    func_name = tool_call["name"]
    func_args = tool_call.get("arguments", {})

    # Execute the tool
    if func_name in tool_functions:
        try:
            result = tool_functions[func_name](**func_args)
        except Exception as e:
            result = f"Error: {e}"
    else:
        result = f"Unknown tool: {func_name}"

    # Build context with tool result and generate natural language response
    call_json = json.dumps({"name": func_name, "arguments": func_args}, separators=(",", ":"))
    full_prompt = tokenizer.decode(prompt_tokens)
    full_context = (
        full_prompt
        + TOOL_CALL_START + call_json + TOOL_CALL_END + END
        + TOOL_RESULT_START + result + TOOL_RESULT_END
        + ASSISTANT
    )
    context_tokens = tokenizer.encode(full_context)

    # Suppress tool tokens in the response — force natural language
    suppress = get_suppress_tokens(tokenizer)

    final_tokens = generate(
        model, context_tokens,
        max_new_tokens=150,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        eos_token_id=end_id,
        suppress_tokens=suppress,
    )

    response_text = tokenizer.decode(final_tokens[len(context_tokens):])
    for marker in ["<|end|>", "<|eos|>", "<|tool_call_start|>", "<|tool_call_end|>"]:
        response_text = response_text.replace(marker, "")
    response_text = response_text.strip()

    return {
        "tool_called": func_name,
        "arguments": func_args,
        "tool_result": result,
        "response": response_text,
    }


def run_direct_response(model, tokenizer, prompt_tokens):
    """Generate a direct text response with tool tokens suppressed."""
    sp = tokenizer.special_tokens
    end_id = sp["<|end|>"]
    suppress = get_suppress_tokens(tokenizer)

    output_tokens = generate(
        model, prompt_tokens,
        max_new_tokens=150,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        eos_token_id=end_id,
        suppress_tokens=suppress,
    )

    gen_text = tokenizer.decode(output_tokens[len(prompt_tokens):])
    for marker in ["<|end|>", "<|eos|>"]:
        gen_text = gen_text.replace(marker, "")
    gen_text = gen_text.strip()

    return {"tool_called": None, "response": gen_text}


def chat(model, tokenizer, user_message, tools, tool_functions, threshold=0.4):
    """Full inference pipeline with smart tool-call routing."""
    system_text = format_tool_schema(tools)
    prompt = f"{BOS}{SYSTEM}{system_text}{END}{USER}{user_message}{END}{ASSISTANT}"
    prompt_tokens = tokenizer.encode(prompt)

    wants_tool, brace_prob = should_call_tool(model, tokenizer, prompt_tokens, threshold)

    if wants_tool:
        result = run_tool_call(model, tokenizer, prompt_tokens, tool_functions)
    else:
        result = run_direct_response(model, tokenizer, prompt_tokens)

    result["brace_prob"] = brace_prob
    return result


# === Fake tool implementations for testing ===
def get_weather(city):
    return f"Sunny, 24C, light breeze in {city}"

def calculate(expression):
    try:
        return str(eval(expression))
    except:
        return "Error evaluating expression"

def get_time(timezone):
    return f"The current time in {timezone} is 2:30 PM"

def web_search(query):
    return f"Top result for '{query}': Wikipedia article about {query}"

def translate(text, target_language):
    return f"Translation to {target_language}: [{text} translated]"

def get_definition(word):
    return f"{word}: a common English word with multiple meanings"

def convert_units(value, from_unit, to_unit):
    return f"{value} {from_unit} = {float(value) * 1.5:.1f} {to_unit} (approximate)"

def set_reminder(message, time):
    return f"Reminder set for {time}: {message}"

def send_message(recipient, message):
    return f"Message sent to {recipient}: {message}"

def tell_story(topic, mood):
    return f"A {mood} story about {topic}: Once upon a time..."


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_time": get_time,
    "web_search": web_search,
    "translate": translate,
    "get_definition": get_definition,
    "convert_units": convert_units,
    "set_reminder": set_reminder,
    "send_message": send_message,
    "tell_story": tell_story,
}

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "return_type": "str",
        "parameters": [{"name": "city", "type": "str"}],
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result.",
        "return_type": "str",
        "parameters": [{"name": "expression", "type": "str"}],
    },
]


def main():
    ckpt_path = os.path.join(PROJECT_DIR, "checkpoints", "sft_best.pt")
    config_path = os.path.join(PROJECT_DIR, "configs", "pebble_25m.yaml")
    tokenizer_path = os.path.join(PROJECT_DIR, "checkpoints", "tokenizer.json")

    model = load_model(ckpt_path, config_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Tokenizer: {len(tokenizer)} tokens\n")

    tools = TOOL_SCHEMAS

    print("=" * 60)
    print("  PEBBLE TOOL-CALLING TEST (Smart Routing)")
    print("=" * 60)

    test_cases = [
        ("What's the weather in Tokyo?", True),
        ("What is 25 * 17?", True),
        ("Hello, how are you?", False),
        ("Tell me the weather for London please.", True),
        ("Tell me a joke", False),
        ("Good morning!", False),
        ("Hi!", False),
        ("Thank you!", False),
        ("What's 144 / 12?", True),
        ("What is 2 + 2?", False),
    ]

    correct = 0
    total = len(test_cases)

    for i, (query, expects_tool) in enumerate(test_cases):
        result = chat(model, tokenizer, query, tools, TOOL_FUNCTIONS, threshold=0.4)
        used_tool = result["tool_called"] is not None
        is_correct = used_tool == expects_tool

        print(f"\n--- Test {i+1} {'PASS' if is_correct else 'FAIL'} ---")
        print(f"User: {query}")
        print(f"P(tool): {result['brace_prob']:.3f}  {'-> TOOL' if used_tool else '-> DIRECT'}")

        if result["tool_called"]:
            print(f"Tool:     {result['tool_called']}({result.get('arguments', {})})")
            print(f"Result:   {result.get('tool_result', '')}")
            print(f"Response: {result['response']}")
        else:
            print(f"Response: {result['response']}")

        if is_correct:
            correct += 1

    print(f"\n{'=' * 60}")
    print(f"  SCORE: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
