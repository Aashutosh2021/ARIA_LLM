# UI Map

ARIA-LLM provides a clean, text-based terminal console interface for interacting with the AI model.

## Console States and Layouts

### 1. Interactive Chat Layout (`chat.py`)
At boot, the interface prints the banner logo, configures device options, and prompts the user:

```text
   _   ___ ___    _        _    _    __  __
  /_\ |_ _| _ \  /_\  ___ | |  | |  |  \/  |
 / _ \ | ||   / / _ \|___|| |__| |__| |\/| |
/_/ \_\___|_|_\/_/ \_\    |____|____|_|  |_|

   ARIA-LLM interactive chat  (type /help for commands)
------------------------------------------------------------
device=cpu  |  vocab=1481  |  params=4,726,016
  mode=sampling  temperature=0.7  top_k=40  top_p=0.95  max_new_tokens=64
------------------------------------------------------------
you> 
```

### 2. Dialogue Streaming
Once a user presses Enter, it prints the assistant reply marker:
```text
you> hello
aira> Hi there! How can I help you today?
you> 
```
Tokens are streamed token-by-token directly to stdout using `print(..., end="", flush=True)` in `chat.py` to give a live typing effect.

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
