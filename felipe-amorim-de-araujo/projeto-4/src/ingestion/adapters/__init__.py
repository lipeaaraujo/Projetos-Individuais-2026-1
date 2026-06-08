from .base import DocumentRef, MziqAdapter, RIAdapter
from .direcional import DirecionalAdapter
from .trisul import TrisulAdapter

ADAPTERS = {
    DirecionalAdapter.name: DirecionalAdapter,
    TrisulAdapter.name: TrisulAdapter,
}


def build_adapters(names, *, user_agent, timeout):
    out = []
    for n in names:
        cls = ADAPTERS.get(n.strip())
        if cls:
            out.append(cls(user_agent=user_agent, timeout=timeout))
    return out


__all__ = ["DocumentRef", "MziqAdapter", "RIAdapter", "ADAPTERS", "build_adapters"]
