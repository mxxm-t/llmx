# `src/inference/media.hpp` - a prompt with images

The one owner of a prompt that holds images ([VISION](../VISION.md), sections 4 and 9), in namespace `infer`.

`open_vision(path, model, tokenizer, dtype, max_pixels)` opens a vision encoder's `mmproj` file beside a text model and keeps it mapped while the encoder reads it, as a `Vision`.
It refuses a model that takes no image rows (`Model::takes_images`), an encoder whose rows are not the width of the model's embedding, an encoder that places its rows other than as a grid (`VisionPositions::grid`, the one placement the model runs) and a vocabulary without the token the encoder's `token_texts()` names for an image's rows.
The encoder runs on the backend of the model's embedding (`Model::embedding_backend`) at the dtype it is given, and `max_pixels` replaces the encoder's own bound on a resized image.

`image_prompt(vision, ids, images)` takes a rendered prompt's ids and the images' file bytes in prompt order.
The prompt must hold exactly one pad id for each image, as HF's processor checks, or it is refused naming both counts.
Each image is decoded, prepared and encoded by `image/` and the encoder one image a call, and its pad id becomes one pad id for each of its rows; the result is an `ImagePrompt`, the expanded ids and a `MediaRun` for each image counted from the prompt's first id, which `Model::prefill` and `Model::score` take beside the ids.
The CLI's `--mmproj` and `--image` call these two and nothing else.
