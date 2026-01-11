# AGENTS.md

## OpenRouter models API
To inspect model capabilities (e.g., `supported_parameters`), query the public Models API:

```bash
curl -s https://openrouter.ai/api/v1/models
```

Filter for a specific model:

```bash
curl -s https://openrouter.ai/api/v1/models \
  | jq -r '.data[] | select(.id=="anthropic/claude-sonnet-4.5") | .supported_parameters'
```

## Python env
Use `uv` for dependency management and syncing the environment.

