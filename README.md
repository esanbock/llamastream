# llamastream

A C++ library that lets you talk to LLMs using iostream-style `<<` and `>>` operators. Send a prompt with `<<`, get the response with `>>` — as naturally as reading and writing to `cin` and `cout`.

```cpp
#include "claudestream.h"

claudestream claude("claude-sonnet-4-20250514", api_key);

std::string response;
claude << "What is the capital of France?";
claude >> response;
// response: "The capital of France is Paris."
```

## Supported Providers

| Class | Provider | Backend |
|---|---|---|
| `claudestream` | [Anthropic Claude API](https://docs.anthropic.com/en/api/messages) | Claude (all models) |
| `bedrockstream` | [AWS Bedrock](https://aws.amazon.com/bedrock/) | Any Bedrock-supported model |
| `ollamastream` | [Ollama](https://ollama.ai/) | Any locally-hosted model |
All providers implement the same `llmstream` base class, so you can swap backends without changing application code:

```cpp
#include "llmstream.h"
#include "ollamastream.h"
#include "claudestream.h"

void ask(llmstream& llm, const std::string& question)
{
    std::string answer;
    llm << question;
    llm >> answer;
    std::cout << answer << std::endl;
}

// Works with any backend
ollamastream ollama("llama3");
ask(ollama, "Explain polymorphism.");

claudestream claude("claude-sonnet-4-20250514", my_api_key);
ask(claude, "Explain polymorphism.");
```

## Building

```bash
mkdir build && cd build
cmake ..
make
```

### Dependencies

- **Required:** libcurl, [nlohmann/json](https://github.com/nlohmann/json)
- **Optional:** AWS SDK for C++ (enables `bedrockstream`)

CMake automatically detects whether the AWS SDK is available and builds Bedrock support if found.

## Provider Setup

### Claude (Anthropic API)

```cpp
claudestream claude("claude-sonnet-4-20250514", "your-api-key");
// Optional: set thinking effort and max tokens
claudestream claude("claude-sonnet-4-20250514", "your-api-key", "low", 4096);
```

### Ollama (local models)

```cpp
ollamastream ollama("llama3");                    // localhost default
ollamastream ollama("llama3", "192.168.1.100");   // remote host
```

### AWS Bedrock

```cpp
bedrockstream bedrock("us.anthropic.claude-sonnet-4-20250514-v1:0");
// Optional: region, temperature, max_tokens, top_p
bedrockstream bedrock("us.anthropic.claude-sonnet-4-20250514-v1:0", "us-east-1", 0.3, 2048, 0.9);
```

Requires AWS credentials configured via environment variables or `~/.aws/credentials`.

## Design

`llmstream` is a pure virtual base class with two operators:

```cpp
class llmstream
{
public:
    virtual llmstream& operator<<(const std::string& prompt) = 0;
    virtual llmstream& operator>>(std::string& response) = 0;
};
```

Each provider implements these operators using its native API. The `<<` operator sends the prompt and stores the result internally; `>>` retrieves it. This keeps the calling code clean and idiomatic C++.

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.
