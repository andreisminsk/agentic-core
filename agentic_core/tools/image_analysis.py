"""image-analysis tool — analyze images using a vision-capable Ollama model."""

import base64
import os

_IMAGE_MIME_TYPES = {
    'image/jpeg', 'image/png', 'image/gif', 'image/bmp',
    'image/webp', 'image/tiff', 'image/x-icon',
}

_MIME_TYPES = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
    '.gif': 'image/gif', '.bmp': 'image/bmp', '.ico': 'image/x-icon',
    '.webp': 'image/webp', '.svg': 'image/svg+xml',
    '.tiff': 'image/tiff', '.tif': 'image/tiff',
}


class ImageAnalysisTool:
    """Analyze an image file using a vision-capable Ollama model.

    The Ollama client is injected at construction time so the core stays
    dependency-free. If no client is provided, the tool lazily imports
    ollama and creates a default client.
    """
    name = "image-analysis"
    do_not_truncate_observations = True
    description = "Analyze an image file using a vision-capable model."
    system_prompt = (
        "## image-analysis\n"
        "Analyzes an image file using a vision-capable Ollama model.\n"
        "Parameters: path (string, required), "
        "prompt (string, optional, default 'Describe this image in detail.').\n"
        "Use /attach-bin <path> to make the model aware of binary files first."
    )
    auto_approve = False  # makes an external LLM call

    def __init__(self, client=None, model=None, base_url=None):
        """Args:
            client: Ollama-compatible client (duck-typed). If None, lazily created.
            model: Vision model name (e.g. 'gemma4:31b-cloud'). If None, uses
                   the same model as the session if available, else a default.
            base_url: Ollama base URL for lazy client creation.
        """
        self._client = client
        self._model = model
        self._base_url = base_url

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from ollama import Client
            self._client = Client(host=self._base_url or "http://localhost:11434")
            return self._client
        except ImportError:
            raise RuntimeError("ollama package not installed")

    def _get_model(self):
        return self._model or "gemma4:31b-cloud"

    def execute(self, params, workdir=None):
        path = params.get("path", params.get("file", ""))
        prompt = params.get("prompt", "Describe this image in detail.")
        if not path:
            return "Error: 'path' parameter is required"
        full_path = os.path.join(workdir or ".", path) if not os.path.isabs(path) else path
        full_path = os.path.normpath(full_path)
        if not os.path.isfile(full_path):
            return f"Error: file not found: {path}"
        ext = os.path.splitext(full_path)[1].lower()
        mime = _MIME_TYPES.get(ext, "")
        if mime not in _IMAGE_MIME_TYPES:
            supported = ", ".join(sorted(_IMAGE_MIME_TYPES))
            return f"Error: not an image ({mime or 'unknown'} for {ext}). Supported: {supported}"
        size = os.path.getsize(full_path)
        if size > 20 * 1024 * 1024:
            return f"Error: image too large ({size / (1024*1024):.1f} MB). Max: 20 MB"
        try:
            with open(full_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return f"Error: failed to read image: {e}"
        try:
            client = self._get_client()
            model = self._get_model()
            response = client.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                }],
            )
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "")
            return response.message.content
        except Exception as e:
            return f"Error: image analysis failed (model: {self._get_model()}): {e}"
