#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "format/gguf.hpp"
#include "image/image.hpp"
#include "model/arch/registry.hpp"
#include "model/encoder.hpp"
#include "model/runtime.hpp"
#include "tokenizer/tokenizer.hpp"

// A prompt with images (docs/VISION.md, sections 4 and 9): a vision encoder's file opened beside a text model.
// A rendered prompt's ids are made into ids with a run of pad ids for each image, and the runs a sequence reads the encoder's rows from.

namespace infer {

// A vision encoder's file opened on the backend of a text model's embedding, its weights held while the encoder reads them, and the preprocessing its images take.
struct Vision {
    gguf::GGUFModel file;
    VisionWeights weights;
    std::unique_ptr<VisionEncoder> encoder;
    image::Params params;
    uint32_t pad = 0;   // the text model's id of the token its template marks an image with
};

// The encoder of `path` beside `model`, whose rows must be the embedding's width and whose pad token the tokenizer must hold; `max_pixels` bounds a resized image where given.
inline std::unique_ptr<Vision> open_vision(const std::string& path, Model& model, const bpe::Tokenizer& tok, backend::Dtype dtype,
                                           std::optional<uint64_t> max_pixels = {}) {
    if (!model.takes_images()) throw std::runtime_error("inference: the model takes no image rows");
    auto v = std::make_unique<Vision>();
    v->file = gguf::read_gguf(path);
    gguf::map_payload(v->file);
    v->weights = vision_weights(v->file);
    v->encoder = std::make_unique<VisionEncoder>(v->weights.arch, v->weights.tensors, model.embedding_backend(), dtype);
    if (v->encoder->plan().out_width != model.embedding_width())
        throw std::runtime_error("inference: the vision encoder's rows are " + std::to_string(v->encoder->plan().out_width) + " wide and the model's embedding " +
                                 std::to_string(model.embedding_width()));
    if (v->weights.arch->positions() != VisionPositions::grid) throw std::runtime_error("inference: the vision encoder places its rows in a way the model does not");
    v->params = v->weights.arch->preprocess();
    if (max_pixels) v->params.max_pixels = *max_pixels;
    const std::string pad = v->weights.arch->token_texts().pad;
    const auto it = tok.token_to_id.find(pad);
    if (it == tok.token_to_id.end()) throw std::runtime_error("inference: the model's vocabulary has no " + pad + " to mark an image with");
    v->pad = it->second;
    return v;
}

// A prompt's ids with each image's pad run, and the runs its sequence reads each image's rows from, counted from the prompt's first id.
struct ImagePrompt {
    std::vector<uint32_t> ids;
    std::vector<MediaRun> runs;
};

// `ids`, a rendered prompt holding one pad id for each of `images`, given as their files' bytes in prompt order: each image decoded, prepared and encoded, and its pad id made one for each of its rows, as HF's processor expands them.
inline ImagePrompt image_prompt(Vision& v, const std::vector<uint32_t>& ids, const std::vector<std::string>& images) {
    size_t marked = 0;
    for (uint32_t id : ids) marked += id == v.pad;
    if (marked != images.size())
        throw std::runtime_error("inference: the prompt marks " + std::to_string(marked) + " images and " + std::to_string(images.size()) + " were given");
    ImagePrompt out;
    size_t next = 0;
    for (uint32_t id : ids) {
        if (id != v.pad) {
            out.ids.push_back(id);
            continue;
        }
        const std::string& bytes = images[next++];
        const image::Prepared prepared = image::prepare(image::decode_image(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()), v.params);
        MediaRun run;
        run.first = out.ids.size();
        run.rows = v.encoder->architecture().tokens(prepared.grid_h, prepared.grid_w);
        run.grid_h = prepared.grid_h / v.params.merge;
        run.grid_w = prepared.grid_w / v.params.merge;
        run.encoded = std::make_shared<const std::vector<float>>(v.encoder->encode(prepared));
        out.ids.insert(out.ids.end(), run.rows, v.pad);
        out.runs.push_back(std::move(run));
    }
    return out;
}

}  // namespace infer
