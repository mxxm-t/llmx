# `src/core/list.hpp` - comma-separated values

`core::comma_list(value)` returns the entries of a comma-separated value in order, empty ones included, so a caller can refuse `a,,b` or a trailing comma with its own message. `backend::device_specs` reads `--device` through it, the CLI reads `--layer-shares` through it, and `llmx-split-check` reads its device list through it.
