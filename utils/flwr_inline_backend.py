from __future__ import annotations

from typing import Callable


def register_inline_backend(name: str = "inline") -> None:
    """Register a lightweight backend to avoid the Ray dependency on Windows."""
    try:
        from flwr.clientapp.client_app import ClientApp
        from flwr.common.context import Context
        from flwr.common.message import Message
        from flwr.server.superlink.fleet.vce.backend import (
            Backend,
            BackendConfig,
            supported_backends,
        )
    except Exception:
        return

    if name in supported_backends:
        return

    class InlineBackend(Backend):  # type: ignore[misc]
        """Simple backend that executes ClientApps sequentially in-process."""

        def __init__(self, _: BackendConfig | None = None) -> None:
            self._app_fn: Callable[[], ClientApp] | None = None

        def build(self, app_fn: Callable[[], ClientApp]) -> None:
            self._app_fn = app_fn

        @property
        def num_workers(self) -> int:
            return 1

        def is_worker_idle(self) -> bool:
            return True

        def terminate(self) -> None:
            return

        def process_message(
            self,
            message: Message,
            context: Context,
        ) -> tuple[Message, Context]:
            if self._app_fn is None:
                raise ValueError("InlineBackend has not been built yet.")
            app = self._app_fn()
            out_message = app(message=message, context=context)
            return out_message, context

    supported_backends[name] = InlineBackend
