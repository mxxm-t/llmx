# `src/inference/media.hpp` - a prompt with images

The one owner of a prompt that holds images ([VISION](../VISION.md), sections 4 and 9), in namespace `infer`.

`open_vision(path, model, tokenizer, dtype, max_pixels)` opens a vision encoder's `mmproj` file beside a text model and keeps it mapped while the encoder reads it, as a `Vision`.
It refuses a model that takes no image rows (`Model::takes_images`), an encoder whose rows are not the width of the model's embedding, an encoder that places its rows other than as a grid (`VisionPositions::grid`, the one placement the model runs) and a vocabulary without the token the encoder's `token_texts()` names for an image's rows.
The encoder runs on the backend of the model's embedding (`Model::embedding_backend`) at the dtype it is given, and `max_pixels` replaces the encoder's own bound on a resized image.

`prepare_image(vision, bytes, identity)` decodes and resizes one image file and gives a `PreparedImage`: its patch grid, its rows, its resized pixels and each stage's time; given the encoder file's identity it also gives the image's digest ([VISION](../VISION.md), section 5), SHA-256 (`sha256_bytes`) over the identity, the preprocessing settings, the grid and the resized pixels.
It runs no backend, so the server prepares an image on the request's connection while the scheduler's thread alone runs the backends.
`ImageEncode(vision, resized)` cuts the resized pixels into patches and begins their encode, holding the encoder (`Vision::encoding`) for its life, and `step()` runs its next piece; `encode_rows(vision, resized)` runs every piece.
`image_tokens(vision, bytes)` gives the rows an image takes from its decoded size alone, which `/v1/tokenize` counts with.
`expand(vision, ids, tokens, images)` makes each pad id of a rendered prompt one for each of its image's rows, as HF's processor expands them, and with `images` a `MediaRun` for each counted from the prompt's first id, which `Model::prefill` and `Model::score` take beside the ids; a prompt marking more or fewer images than were given is refused naming both counts, as HF's processor refuses it.
`expand` leaves the runs' rows to the caller: `image_prompt(vision, ids, images)`, the CLI's, prepares each image file without a digest, expands and encodes; the server prepares with the digest, expands, and gives the scheduler an `ImageEncode` to step where a sequence reads an image's rows.
