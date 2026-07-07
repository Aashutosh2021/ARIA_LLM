# UI Map

ARIA-LLM provides a clean, text-based terminal console interface for interacting with the AI model.

## Console States and Layouts

### 1. Interactive Chat Layout (`chat_qwen.py`)
At boot, the interface prints the banner logo, configures device options, and prompts the user:

```text
   _   ___ ___    _
  /_\ |_ _| _ \  /_\    from-scratch architecture
 / _ \ | ||   / / _ \   running real Qwen2.5 weights
/_/ \_\___|_|_\/_/ \_\

Loading tokenizer + converted Qwen weights ...
Ready on cuda (NVIDIA GeForce RTX 3050 Laptop GPU).  Params: 486,177,280  (our architecture, Qwen weights)
Type /help for commands.
----------------------------------------------------------------
you> 
```

### 2. Dialogue Streaming
Once a user presses Enter, it prints the assistant reply marker:
```text
you> hello
aira> Hi there! How can I help you today?
you> 
```
Tokens are streamed token-by-token directly to stdout using `print(..., end="", flush=True)` in `chat_qwen.py` to give a live typing effect.

### 3. Help Docs Panel (`/help`)
Typing `/help` prints the class docstring:
```text
you> /help
ARIA-LLM
Real Conversation on the From-Scratch Architecture

This is the headline demo of the project.
...
In-chat commands:
    /reset             clear the conversation history
    /system <text>     set a new system prompt and reset
    /temp <value>      change sampling temperature
    /help              show this help
    /exit  /quit       leave
```
