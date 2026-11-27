# `src/model/arch/inkling2/` - Inkling-2-Mini (`inkling2_loop`)

Inkling-2-Mini's module, built in the order `docs/INKLING2.md` (section 7) gives; today its configuration and its plan, which the registry reaches and which refuses the architecture by name until the CPU ops land.

- `config.hpp`: `read_config(file)` reads the `inkling2_loop.*` keys into a `Config`, every key required and refused as missing by its name; the values the model assumes are refused as unsupported when they differ: `loop.varying` true, a nonzero `rope.dimension_count` (the model places tokens by its relative bias and short convolutions, with no rotary positions), `shortconv.l_cache` other than 4, an expert width other than the feed-forward width, and value widths other than the key widths.
  `check_config` refuses per-layer head counts or a window pattern that do not cover the layers, query heads that are not whole groups of a layer's KV heads, an unpadded vocabulary above the padded one, more experts a token than a layer has, a loop outside the layers, and a loop that does not start and end on attention-residual blocks, spans fewer than two blocks or sits in a model of more than four blocks, which the six-row residual bundle could not carry (docs/INKLING2.md, section 3).
  `execution_layers(config)` is the physical layer each execution layer runs: those before the loop, the loop `loop.count` times, then the rest, 24 on the fixture and 48 on the real file.
- `model.hpp`: `Inkling2`, opened by `open(file, prefix, request)` from the registry's `inkling2_loop` entry.
  `layout(tensors)` plans every execution layer with its physical layer's roles, each pass of the loop repeating the same names and role ids, and its kind flags (full attention, block boundary, second or third pass, the loop's first layer, a block query read, in the loop).
  The head reads the first `vocab_size_unpadded` rows of the padded `output.weight`, and `plan.vocab` is the unpadded vocabulary; the residual is the six-row bundle.
  `embed` writes a fed row's bundle row 0 from the id a device holds (`embed_ids`, `embeds_fed`), over the placeholder's row, before the embedding norm.
  Every layer keeps entries with the window as headroom: a sliding layer its K and V rings of twice the window and its convolution rings of three taps plus the window, a full-attention layer the convolution rings and one pooled K and V row a position.
  A tensor a layer's kind does not read (a full-attention tensor on a sliding layer or the reverse, a block query on layer 0, a loop tensor outside the loop) is refused, and every role's tensor must exist.
  `plan(tensors)` runs the layout, then refuses with "inference: unsupported architecture inkling2_loop, whose math is not implemented yet", as do the math parts.
