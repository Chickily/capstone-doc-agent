GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLM:
    """The only file that talks to an LLM provider.

    provider="gemini"    -> Google AI Studio (free tier) through its OpenAI-compatible endpoint
    provider="anthropic" -> Anthropic Claude API
    Both expose the same method: complete(system, messages) -> text.
    """

    def __init__(self, api_key: str, model: str, provider: str = "gemini"):
        if not api_key:
            raise ValueError("No API key configured.")
        self.model, self.provider = model, provider
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key, max_retries=3, timeout=90)
        elif provider == "gemini":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL,
                                 max_retries=3, timeout=90)
        else:
            raise ValueError(f"Unknown LLM provider '{provider}'. Use 'gemini' or 'anthropic'.")

    def complete(self, system: str, messages: list, max_tokens: int = 1024) -> str:
        """messages = [{"role": "user"|"assistant", "content": "text"}, ...]"""
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model, system=system, messages=messages, max_tokens=max_tokens)
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return (resp.choices[0].message.content or "").strip()
