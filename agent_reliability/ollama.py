"""Opt-in loopback Ollama HTTP adapter. Nothing calls it during replay."""
import json
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from .contracts import ContractError, parse_action, strict_json


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ContractError("redirects prohibited")


class OllamaAdapter:
    def __init__(self, model: str, endpoint="http://127.0.0.1:11434", transport=None):
        address = urlsplit(endpoint)
        if (address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost", "::1"}
                or address.username or address.password or address.query or address.fragment
                or address.path not in {"", "/"}):
            raise ValueError("only credential-free loopback HTTP origins are allowed")
        if not model or len(model) > 200:
            raise ValueError("explicit local model required")
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.transport = transport or self._http
        self.calls = 0

    def _http(self, request):
        # Ignore ambient proxy settings; never send this prompt to a proxy or redirect.
        opener = build_opener(ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=30) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ContractError("oversized HTTP response")
        return raw.decode("utf-8")

    def propose(self, public_task, version, step):
        # Whitelist public fields: caller-added secrets never become prompt content.
        clean = {"task_id": public_task["task_id"], "increments": public_task["increments"]}
        prompt = {"task": clean, "step": step, "version": version,
                  "instruction": "Return only JSON with integer step, amount, version for this task step."}
        payload = {"model": self.model, "stream": False, "format": "json",
                   "prompt": json.dumps(prompt), "options": {"temperature": 0}}
        request = Request(self.endpoint + "/api/generate", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        self.calls += 1  # Includes failed HTTP attempts.
        envelope = strict_json(self.transport(request))
        if envelope.get("done") is not True or type(envelope.get("response")) is not str:
            raise ContractError("incomplete Ollama generation")
        return parse_action(envelope["response"], clean, version, step)
