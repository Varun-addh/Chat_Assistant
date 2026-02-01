"""Global warnings filters for clean logs.

Python automatically imports `sitecustomize` on startup (if present on sys.path),
which makes this a safe place to silence known third-party deprecation noise.

Keep filters narrowly-scoped to specific messages.
"""

import warnings

# pkg_resources google namespace warning (emitted by some google-* packages)
warnings.filterwarnings(
    "ignore",
    message=r"Deprecated call to `pkg_resources\.declare_namespace\('google'\)`\..*",
)

# transformers/torch warning about deprecated pytree registration helper
warnings.filterwarnings(
    "ignore",
    message=r"torch\.utils\._pytree\._register_pytree_node is deprecated\..*",
)
