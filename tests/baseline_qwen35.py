import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import common
from common import device_args, require
from baseline_8b import file_sha256
from qwen35 import refusal
import spec_decode

# The qwen35 family's real-model check: llmx on a pinned Qwen3.5 file against the HF goldens of its model, which tools/gen_baseline.py qwen35 made with HF's full forward in float32.
# tests/baseline.py runs the gate's qwen35 files at 512-token windows, the 4096-token windows and files outside the gate run by hand through main(), and every checkpoint shares one tokenizer.json.

DATA = Path(__file__).resolve().parent / "data"
TOKENIZER_GOLDEN = "baseline_tokenizer_qwen35.json"
VOCAB_SIZE = 248320
MODEL_CONTEXT = 262144
COMMAND_TIMEOUT = 3600
PPL_GOLDENS = {512: "baseline_perplexity.json", 4096: "baseline_perplexity_4096.json"}

# Every golden this check reads, with the SHA-256 of its text with LF line endings.
GOLDEN_SHA256 = {
    "baseline_tokenizer_qwen35.json": "795c8c9fd839a542d17ed8826a1322dbbd78e107fb12ab969587c252dcde8d67",
    "qwen35-0.8b/baseline_logits.json": "faa5d386b9eb0842e3f01f442fdd5478a744026d5606e6cff11f302a9ec3525b",
    "qwen35-0.8b/baseline_chat.json": "9d0c6903633934efcc260ae38de4185f99c10824801957b11efa4fbd6acba3b1",
    "qwen35-0.8b/baseline_perplexity.json": "42c5ef92ec3da19b8ef67b81aa5cfafd55b7c9a9413a76c4e65df9a57d8e98f4",
    "qwen35-0.8b/baseline_perplexity_4096.json": "6c8bd8601709199c5ddde46d637ece9242efe03222a403a808ab0dca0a6c789a",
    "qwen35-4b/baseline_logits.json": "2a2fbb27a70a6344835e751432bcc5a47aae73e542cb1e11562de09a4ef7d361",
    "qwen35-4b/baseline_chat.json": "0fc52ab64ddb855c00bd441a702d91d7fb2c40bcafa6320636baf21740f5bab8",
    "qwen35-4b/baseline_perplexity.json": "bea1ad500f60333c546c328523efb0d9edf24deb7d8f99decb1b6e67e3359cbb",
    "qwen35-4b/baseline_perplexity_4096.json": "e19d2510d6acccb10fb292c530899e47843d7caff97915e1a93cb9d544af01e4",
    "qwen3.6-27b-fp8/baseline_logits.json": "54bf19ffe11c35713bfe41d97da981b7d1251b42f1736f4bfb512aba36d3d30a",
    "qwen3.6-27b-fp8/baseline_perplexity.json": "d11a954b136fe7b91ce367d9957eaf6948f23d9aedd7811429b2c892221651ae",
    "qwen35-0.8b-q4_k_m-file-exact/baseline_logits.json": "28535cbbebbcf561e90a7e92ae57edacaae479d29a5e14d4d7fc4b8ac8014720",
    "qwen35-0.8b-q4_k_m-file-exact/baseline_perplexity.json": "d723cbe8fc1a2eff5cac8b81d35dd0fa03ffeda182ceaf86b2fa82e38c9aafa7",
    "qwen35-0.8b-q8_0-file-exact/baseline_chat.json": "189b51585c63e0891419bc176d635d0093635621fa95ac3ed273b531ec4edd92",
    "qwen35-0.8b-q8_0-file-exact/baseline_logits.json": "81a2b68279d33607144730d64641e793d885fd7ba0740aeabbc202dd99d026ff",
    "qwen35-0.8b-q8_0-file-exact/baseline_perplexity.json": "49dadc76a5c8c30fd3b1d2316074b5da7641e380db31ae9745670d4646c8ba1f",
}

# Each file's bounds against its model's goldens, set from llmx's first measurement on it plus a margin (docs/ASSETS.md); a file without them is measured and fails.
# Every top-1 must be HF's.
BOUNDS = {
    "Qwen3.5-0.8B-Q8_0.gguf": {"top5_overlap": 5, "continuous_nll": 0.02, "window_nll": 0.02},
    "Qwen3.5-4B-Q4_K_M.gguf": {"top5_overlap": 4, "continuous_nll": 0.07, "window_nll": 0.08},
}
# A file against its file-exact goldens is held to this file's bounds whatever its type, since the format's loss is on both sides.
FILE_EXACT_BOUNDS = "Qwen3.5-0.8B-Q8_0.gguf"

# The Qwen3.5-0.8B Q4_K_M, whose own quantization moves HF's top-1 on two of its eight rankings (docs/CI.md).
# Its correctness gate is its committed file-exact goldens at FILE_EXACT_BOUNDS, and against its model's goldens it is held to its own quantization's cost, for this file's SHA-256 alone (docs/ASSETS.md).
QWEN35_08B_Q4_K_M_SHA256 = "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"
QWEN35_08B_Q4_K_M_QUALITY = {"top1_matches": 6, "top5_overlap": 4, "continuous_nll": 0.021, "window_nll": 0.025}
# Each file's committed file-exact goldens, by its SHA-256, a directory under tests/data whose goldens GOLDEN_SHA256 pins.
QWEN35_08B_Q8_0_SHA256 = "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c"
FILE_EXACT = {QWEN35_08B_Q4_K_M_SHA256: "qwen35-0.8b-q4_k_m-file-exact", QWEN35_08B_Q8_0_SHA256: "qwen35-0.8b-q8_0-file-exact"}
# A Hugging Face directory of a model's goldens' own reference weights, by its goldens' directory and the SHA-256 of each weight file (docs/ASSETS.md): the goldens are file-exact for it, so it is held to FILE_EXACT_BOUNDS.
DIRECTORIES = {
    "qwen35-0.8b": {"model.safetensors-00001-of-00001.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"},
    "qwen35-4b": {"model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
                  "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"},
    # Qwen/Qwen3.6-27B-FP8 a56219b, its FP8 weights read as tests/spec_decode.py decodes them, times their block scales.
    "qwen3.6-27b-fp8": {
        "layers-0.safetensors": "5a6c052e37a754549e8f81c6fb32b050419f0fc2f71598817e08d0446b3f309d",
        "layers-1.safetensors": "9a2efd1048386e560c8241c7b972e213b516f0cbeafc4de6f486bd7faddb8c39",
        "layers-2.safetensors": "c2709097f77e8ec7204f908f54d7b84343139163e600815789124cbf15a709c6",
        "layers-3.safetensors": "a874cf17d3480894aa13eb34568e1025f10c4d6b96f483cd93be583849312a0f",
        "layers-4.safetensors": "9eb31d85023776894347a56101ee067f369d0d3c024f1331f5c7cba1f45b6df4",
        "layers-5.safetensors": "a8c9ea40638fd66bd8ab21940df215e7921b4af04646d758d74425be011a9b6f",
        "layers-6.safetensors": "ad8b704be0cde8a774c4a49af59336d2b48b574446c1eddff8ba35d20d05ae69",
        "layers-7.safetensors": "2acb7e8d035d730a1c81c36c3b20ca15020b89d2bcd4462259f2ecb2a8dcb828",
        "layers-8.safetensors": "81ebe095241562b2e2ace8875e2771f0cdb68e76286a18506ea19a749b8de9b4",
        "layers-9.safetensors": "fdd926f3e8c335eb21a2b41cf635cb241f337a5d47310ef41ae3d43bf53410c3",
        "layers-10.safetensors": "ebc4fd84671c2ac2d8d7b8da3707ed5d30738771a22eecc883d707521ed513a0",
        "layers-11.safetensors": "9daf0f0b763489c0869061669293f93213d6060b1b0c88061b53bb97f9082cd6",
        "layers-12.safetensors": "922729fcaedaece39e7b0fba137e3b900fbc3fe6bd8a85b16a215658be86938f",
        "layers-13.safetensors": "61c0a820b09d58e45e516c56532007155b0fd214c11fa8906640c9217da95c84",
        "layers-14.safetensors": "c2faab3a2ecddde0e762b3870c75f6fd83886edbabbd99ac40d641122d29e652",
        "layers-15.safetensors": "e75fbe4de60c0420912b91f379c29c95a49644feee6fe8a0bd3dd638ae07ea2b",
        "layers-16.safetensors": "84776e1d0a62c40730a4bc0f0e2c68242ba9a6150975d8992747fb6ecc0ebfcc",
        "layers-17.safetensors": "c7e32717b97f9553d62734f01bed4cc970dcf6bd3d5ba67c7704290656523d7e",
        "layers-18.safetensors": "1c6aed5f416dee34bb7ccfc322e0aa33f192b40afeab4b69fce784b2b349e253",
        "layers-19.safetensors": "63e1b8af41de9851ed279f9d4f42566febf81d13ae47e469ea1009a26c6b64e0",
        "layers-20.safetensors": "c5945b2e40f3d97cb01d0cedcb60ea36f61ab585179ab0c02fdeec3d360f724d",
        "layers-21.safetensors": "c8a7696ad5cd1be016cfc553351ee4b588f3d972084dd47028db8dba9b113870",
        "layers-22.safetensors": "08481a6a46057d0bcdcd2598d4f67d5f821725fc483800cd360c2d7047907e08",
        "layers-23.safetensors": "13b2ba6536c77330fa1c40e91c4c707d5086803299e0f57dd03142bb28faaaa4",
        "layers-24.safetensors": "0f7855ee6b018e707d015f445f7e8b9542003104dbd0436e83e8c0dec73bdd17",
        "layers-25.safetensors": "fe49c034e095263e51b391895ccb5098741e06d75a6788769beb72fd50708846",
        "layers-26.safetensors": "14f086ff505e9df89c2fb938c947b0d023f6f19d3cecdc7f03a70caf36ea6ed4",
        "layers-27.safetensors": "4fa89a43e7196b18e6c67478aeedc1b3ab4853fa3bc4305d4ac73341b70e7e24",
        "layers-28.safetensors": "f10aecd4f9800052bbf1ad2cca939eb44dc205d42ff546096ce9212bd6eae411",
        "layers-29.safetensors": "633058fe6f1ec434c79e5c50d04c83b1955d52ca1889b1adca0e2db1cad9fa16",
        "layers-30.safetensors": "f9af47cefbdb300129c25eee1127c50629d9a2eba7f862a58a430a2cb378cf55",
        "layers-31.safetensors": "2a2e87f569b5da06a87b83df628cb27f491f3dbed55ca82054820cb382cffd2e",
        "layers-32.safetensors": "ca63c9dc522a1f0efd39e2cd7f5fb0efdb4274eabce5c15cc9aab11463cb3bb8",
        "layers-33.safetensors": "bb8eb220efa6a599a4ee29914d59ca26fe13fe64ec6f9a5ccff300432abc572b",
        "layers-34.safetensors": "6e08147ff5c8e0d5d7207d62ede78d6194f1453165cfab940743ff327bb0e6e5",
        "layers-35.safetensors": "f4391ffd8245e648eeaad69954721f106ea0423a56f9abee921d3d2880d98c03",
        "layers-36.safetensors": "5b077d63785d5c2b20ab5c0b8895b08941af08bb5685a1b6a5a3292c3eac638f",
        "layers-37.safetensors": "d38c28d8c86cc147e5edc1935dec46687a6b3a8bc0d712e829003dd9cd4fab7d",
        "layers-38.safetensors": "1628a41fa5d3dbcf2ffde423530a9e94e7228e247e922e2269bd348071a76365",
        "layers-39.safetensors": "889c827bbff3fa6b78558dcdfd24670894b8a450395ec418e7a4f3202789e0b1",
        "layers-40.safetensors": "5f90d2b2d16e5c51a078f21eed843d80c22af71401669577d274ea7f2c1ce95f",
        "layers-41.safetensors": "80877a16835f7311a112ad0bbaf1d19997142ea3e1df95ffe446fd52ad1f80d8",
        "layers-42.safetensors": "c1bf34d1a3e4d36b342c5c680bb12c4ce74f02530dff33b735fad27470ef2594",
        "layers-43.safetensors": "10eceff499a24e0072a40b960671f00fc5012d7451f22cb6883f358e085ce276",
        "layers-44.safetensors": "673543b94e59003265ae648bba11389c6735ee3264ec164108b6db99f1219587",
        "layers-45.safetensors": "77b8c7a9cc451a378ee8d6927ce1c4b9664adb83c4786440eec099591dc98f32",
        "layers-46.safetensors": "9ee06b8029b7812644bd5fb01b8bfea7b640353769d094ae59d8c82d986a8b7b",
        "layers-47.safetensors": "6a8e0ddb389485965cd30bd0a90d1be5e1acb65cb6311ce337199f268c3120f6",
        "layers-48.safetensors": "70bd74dcdf4ea8b09e1ae3463977a05aba73f2c02397491bb641a0a3ca6832a3",
        "layers-49.safetensors": "a3481f971433ecc9041f44a555e443ff124e037601b36eab0bb3057946e041e4",
        "layers-50.safetensors": "1aa60d9cb7f867a2033be0b1ff053049f09a6623826dd0d37daed3281479b875",
        "layers-51.safetensors": "511f64772c1a5a15292eb1c48166b9d206088aed283185e10877fc43308097bd",
        "layers-52.safetensors": "92c35a4e850b4c0b87a1eab42c47996504e062a8a11699ada20fcea0b88f36c7",
        "layers-53.safetensors": "ab48cad204502ddbf17f5ee2d5e52457baac965300cc15e8a25f35c843b4d526",
        "layers-54.safetensors": "2804cee95e18e9e516b9490c36ea7f41553a3fcb4455cf44a361d6086ed214a3",
        "layers-55.safetensors": "543841981255fce51f0d84fe2db4de580594f6c6d6a590ae47ffbdbd01211571",
        "layers-56.safetensors": "9847da2610e67b1f0ff39d8d103c7814f95bb8ede4204d5ae3055a8589bb0313",
        "layers-57.safetensors": "32fe173c033e46648bc71bc8e5036565234c0c13ac66acb2ad84c9cbb426b90e",
        "layers-58.safetensors": "c9858a0c4bd8cb980829a45d3f9ca52a0bd506c6a0775eb179d3e5ebfaae7867",
        "layers-59.safetensors": "3c30fd14e883c15af50e2f647c2d0b8f3a2fca96904e4b37cc8bb1956a133606",
        "layers-60.safetensors": "1f24aa7a99e72f658089462b258613b11d06d6e11733ff5d5f864c663a190f92",
        "layers-61.safetensors": "0b083f4afca3538f87545d4fc50817292fc80061a731a5f0235f5c406be87b45",
        "layers-62.safetensors": "b1e4c80f95241cc7d269d7cf6a35291c8b455f2d4f1381e4eafbfccb35add125",
        "layers-63.safetensors": "7e27ec71df803eb38ba93aead54daaf1110d3f7124416443a30a3c81abc1c805",
        "mtp.safetensors": "9557770331f1f648eb96039f2a1e7cdc5742fe15c0f5777c43063b2c12a60f4c",
        "outside.safetensors": "27a91100d904f6acc1c86d08675691199e1fe2da5da613106fb01ae4809b3ac1"},
}
MAX_PLAUSIBLE_LOGIT = 100.0
# What a file without bounds is measured at: every rule of the validators except the bounds themselves.
MEASURE_ONLY = {"top5_overlap": 0, "max_abs_logit": MAX_PLAUSIBLE_LOGIT, "continuous_nll": math.inf, "window_nll": math.inf}


def ids_sha256(ids):
    """The digest a golden keeps in place of a long id list: SHA-256 of the ids in decimal, joined by commas."""
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def wiki_excerpt(chars):
    """The first `chars` characters of tests/data/wiki.test.raw, its line endings read as LF, as every perplexity golden takes its text."""
    with open(DATA / "wiki.test.raw", encoding="utf-8") as f:
        return f.read(chars)


def load_golden(name):
    """The golden `name`, a path under tests/data, refused unless its text has the pinned SHA-256."""
    # Git may check JSON out with CRLF; the generator writes LF.
    text = (DATA / name).read_text(encoding="utf-8")
    require(name in GOLDEN_SHA256 and hashlib.sha256(text.encode("utf-8")).hexdigest() == GOLDEN_SHA256[name],
            "qwen35 golden changed or not pinned: " + name)
    return json.loads(text)


def goldens_for(file):
    """The directory of the model goldens made for the pinned file `file`, which lists it in each of them, and those goldens by name."""
    for directory in sorted({name.split("/")[0] for name in GOLDEN_SHA256 if "/" in name} - set(FILE_EXACT.values())):
        docs = {name.split("/")[1]: load_golden(name) for name in GOLDEN_SHA256 if name.startswith(directory + "/")}
        if all(file in doc.get("gguf_files", ()) for doc in docs.values()):
            return directory, docs
        require(not any(file in doc.get("gguf_files", ()) for doc in docs.values()), "%s lists %s in only some of its goldens" % (directory, file))
    raise ValueError("no qwen35 goldens are made for " + file)


def check_chat_template(model, expected):
    """The chat template in the file `model` must be the one the chat golden was rendered with."""
    template = spec_decode.GGUF(model).value("tokenizer.chat_template")
    require(template is not None, "the model file has no chat template")
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    require(digest == expected, "the file's chat template has SHA-256 %s, the golden's %s" % (digest, expected))
    return {"sha256": digest}


def ppl_text(doc):
    text = doc["text"] if "text" in doc else wiki_excerpt(doc["chars"])
    require(hashlib.sha256(text.encode("utf-8")).hexdigest() == doc["text_sha256"], "PPL text hash differs")
    return text


def check_ids_digest(output, n_tokens, digest):
    ids = common.parse_ids(output)
    require(len(ids) == n_tokens and ids_sha256(ids) == digest, "token IDs differ from HF: %d tokens, SHA-256 %s" % (len(ids), ids_sha256(ids)))
    return {"tokens": len(ids)}


def bounds_for(spec):
    """The bounds the pinned file `spec` is held to against its model's goldens: the 0.8B Q4_K_M's own for that file's SHA-256, else its entry in BOUNDS, None until they are set."""
    if spec["sha256"] == QWEN35_08B_Q4_K_M_SHA256:
        return QWEN35_08B_Q4_K_M_QUALITY
    return BOUNDS.get(spec["file"])


def file_exact_goldens(spec, directory=None):
    """The logit, chat where made, and 512-token perplexity goldens tools/gen_baseline.py file-exact made from the pinned file `spec` itself: those in `directory`, or the committed ones of FILE_EXACT, pinned by SHA-256; None where it has none."""
    names = ("baseline_logits.json", PPL_GOLDENS[512])
    if directory:
        names += ("baseline_chat.json",) if Path(directory, "baseline_chat.json").exists() else ()
        docs = {name: json.loads(Path(directory, name).read_text(encoding="utf-8")) for name in names}
    elif spec["sha256"] in FILE_EXACT:
        names += ("baseline_chat.json",) if FILE_EXACT[spec["sha256"]] + "/baseline_chat.json" in GOLDEN_SHA256 else ()
        docs = {name: load_golden(FILE_EXACT[spec["sha256"]] + "/" + name) for name in names}
    else:
        return None
    for doc in docs.values():
        require(doc.get("weights", {}).get("sha256") == spec["sha256"], "the goldens in %s were not made from %s" % (directory or FILE_EXACT[spec["sha256"]], spec["file"]))
        require("activation_rounding" not in doc, "%s holds an HF control with rounded inputs, not goldens" % (directory or FILE_EXACT[spec["sha256"]]))
    return docs


def model_goldens(model, file_exact=None):
    """The goldens the pinned qwen35 file `model` is held to, by name, and their bounds, None until they are set.
    They are its model's goldens, or with `file_exact` the file-exact goldens in that directory, at the Q8_0 file's bounds whatever the file's type."""
    spec = pinned(model)
    if not file_exact:
        return goldens_for(spec["file"])[1], bounds_for(spec)
    return file_exact_goldens(spec, file_exact), BOUNDS.get(FILE_EXACT_BOUNDS)


def check_file(model, context, run, check, write, docs, bounds):
    """Every check of the pinned qwen35 file `model` against the goldens `docs` at `bounds`, with perplexity windows of `context` tokens.
    `run(label, arguments)` runs llmx and gives (exit code, output), `check(label, validator, *arguments)` records one check, and `write(name, data)` keeps a file beside the results and gives its path.
    Every ranking's top-1 must be HF's, unless `bounds` names how many a file keeps (`top1_matches`), which one more check counts over them all.
    Returns "skip" when llmx does not run the architecture here (qwen35.REFUSALS), "unbounded" when there are no bounds yet, which measures every check at MEASURE_ONLY, and "checked" otherwise."""
    limits = dict(bounds or MEASURE_ONLY, max_abs_logit=MAX_PLAUSIBLE_LOGIT)
    top1_needed = limits.get("top1_matches")

    def output(label, arguments):
        rc, out = run(label, arguments)
        require(rc is not None, "%s timed out after %d s" % (label, COMMAND_TIMEOUT))
        require(rc == 0, "%s failed (exit %d): %s" % (label, rc, out.strip()[-200:]))
        return out

    for index, case in enumerate(load_golden(TOKENIZER_GOLDEN)["cases"]):
        label = "tokenizer-%02d" % index
        check(label, common.check_ids, output(label, ["tokenize", model, case["text"]]), case["ids"])
    chats = []
    if "baseline_chat.json" in docs:
        # A directory carries its repository's template, not the GGUF file's the chat goldens were rendered with; their cases are rendered text either way.
        if not Path(model).is_dir():
            check("chat-template", check_chat_template, model, docs["baseline_chat.json"]["template_sha256"])
        chats = docs["baseline_chat.json"]["cases"]
    for index, case in enumerate(chats):
        label = "chat-%02d-ids" % index
        check(label, common.check_ids, output(label, ["tokenize", model, case["text"]]), case["token_ids"])
    for index, case in enumerate(docs["baseline_logits.json"]["cases"]):
        label = "logits-%02d-ids" % index
        check(label, common.check_ids, output(label, ["tokenize", model, case["text"]]), case["token_ids"])
    doc = docs[PPL_GOLDENS[context]]
    text = ppl_text(doc)
    check("ppl-ids", check_ids_digest, output("ppl-ids", ["tokenize", model, text]), doc["n_tokens"], doc["token_ids_sha256"] if "token_ids_sha256" in doc else ids_sha256(doc["token_ids"]))

    prompts = [("logits-%02d" % i, case, ["logits", model, case["text"]]) for i, case in enumerate(docs["baseline_logits.json"]["cases"])]
    for index, case in enumerate(chats):
        prompts.append(("chat-%02d" % index, case, ["logits", model, "--file", write("chat-%02d.txt" % index, case["text"].encode("utf-8"))]))
    matched = 0
    for label, case, arguments in prompts:
        rc, out = run(label, arguments + ["--top", "10", "--threads", "6"])
        if refusal(rc, out):
            return "skip"
        require(rc is not None, "%s timed out after %d s" % (label, COMMAND_TIMEOUT))
        require(rc == 0, "%s failed (exit %d): %s" % (label, rc, out.strip()[-200:]))
        check(label, common.check_logits, out, case, VOCAB_SIZE, limits, top1_needed is None)
        matched += common.first_id(out) == case["top_ids"][0]
    if top1_needed is not None:
        check("top1", common.check_top1, matched, len(prompts), top1_needed)
    excerpt = write("excerpt.txt", text.encode("utf-8"))
    for index, case in enumerate(common.ppl_cases(doc)):
        for mode in common.PPL_MODES:
            label = "ppl-%02d" % index + ("-per-token" if mode == "per-token" else "")
            check(label, common.check_ppl, output(label, common.ppl_command(model, excerpt, case, mode)), case, doc["n_tokens"], MODEL_CONTEXT, limits)
    return "checked" if bounds else "unbounded"


def directory_goldens(model):
    """The goldens of the pinned directory `model`, by name, and the SHA-256 of each weight file; the goldens' reference revision is the directory's, whose weights they ran in float32."""
    for golden, weights in DIRECTORIES.items():
        if all(Path(model, file).is_file() for file in weights) and all(file_sha256(Path(model, file)) == digest for file, digest in weights.items()):
            return {name.split("/")[1]: load_golden(name) for name in GOLDEN_SHA256 if name.startswith(golden + "/")}, weights
    require(False, "%s is not a pinned qwen35 directory" % model)


def check_pinned(model, context, run, check, write, file_exact=None):
    """Every check of the pinned qwen35 file `model` as check_file runs them: against the file-exact goldens in `file_exact` when given, and otherwise against its model's goldens, after its committed file-exact goldens where it has them.
    `check` returns whether its check passed. A file whose committed file-exact goldens fail is not held to its model's goldens, so it cannot pass on its quality bounds; the file-exact checks' labels start with "file-exact-".
    Returns check_file's status, or "failed" when the committed file-exact goldens failed."""
    spec = pinned(model)
    if file_exact:
        return check_file(model, 512, run, check, write, file_exact_goldens(spec, file_exact), BOUNDS.get(FILE_EXACT_BOUNDS))
    committed = file_exact_goldens(spec)
    if committed:
        failed = []

        def exact(label, validator, *arguments):
            passed = check("file-exact-" + label, validator, *arguments)
            if not passed:
                failed.append(label)
            return passed

        status = check_file(model, 512, run, exact, write, committed, BOUNDS.get(FILE_EXACT_BOUNDS))
        if status == "skip" or failed:
            return status if status == "skip" else "failed"
    return check_file(model, context, run, check, write, goldens_for(spec["file"])[1], bounds_for(spec))


def pinned(model):
    """The entry of tests/data/fixtures.json for the qwen35 file `model`, refused unless the file has its SHA-256."""
    from baseline import pinned_fixture
    spec = pinned_fixture(os.path.basename(model))
    require(spec is not None and spec["family"] == "qwen35", "%s is not a pinned qwen35 fixture" % model)
    require(file_sha256(Path(model)) == spec["sha256"], "%s does not have the pinned SHA-256" % model)
    return spec


def run_hosted():
    """The 512-token check of every qwen35 fixture of the gate on disk, which the hosted HF job downloads, in the suite's form: a line for each file, False on the first failure, and common.SKIPPED when no file was on disk."""
    import tempfile
    from baseline import PINNED, snapshot_path
    ran = 0
    for spec in (s for s in PINNED if s["family"] == "qwen35" and s["gate"]):
        name = "baseline-qwen35[%s]" % spec["file"]
        if os.environ.get("LLMX_BASELINE_GGUF"):
            print("%s: SKIP - LLMX_BASELINE_GGUF selects one Qwen3 fixture" % name)
            continue
        model = snapshot_path(spec["repo"], spec["revision"], spec["file"])
        model = str(model) if model.exists() else None
        if not model:
            print("%s: SKIP - fixture model not on disk" % name)
            continue
        ran += 1
        failures, count = [], [0]

        def run(label, arguments):
            return common.run(arguments)

        def check(label, validator, *arguments):
            count[0] += 1
            try:
                validator(*arguments)
                return True
            except (ValueError, OverflowError) as error:
                failures.append("%s: %s" % (label, error))
                return False

        with tempfile.TemporaryDirectory(prefix="llmx_qwen35_") as directory:
            def write(file, data):
                path = os.path.join(directory, file)
                with open(path, "wb") as f:
                    f.write(data)
                return path

            try:
                status = check_pinned(model, 512, run, check, write)
            except ValueError as error:
                failures.append(str(error))
                status = "failed"
        if failures:
            print("%s: %d of %d checks fail:" % (name, len(failures), count[0]))
            for failure in failures:
                print("    " + failure)
            return False
        if status == "skip":
            print("%s: SKIP - llmx does not run the qwen35 architecture here (%d tokenizer and chat id checks pass)" % (name, count[0]))
        elif status == "unbounded":
            print("%s: FAIL - no bounds in tests/baseline_qwen35.py; set them from `tests/baseline_qwen35.py --model` on this file" % name)
            return False
        else:
            print("%s: %d checks  [ok]" % (name, count[0]))
    return True if ran else common.SKIPPED


def main(argv=None):
    global COMMAND_TIMEOUT
    parser = argparse.ArgumentParser(description="The qwen35 real-model HF check of one pinned file, with a report; no downloads.")
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="a pinned qwen35 file, or a pinned Hugging Face directory of its model (DIRECTORIES)")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for raw output and report")
    parser.add_argument("--context", type=int, choices=sorted(PPL_GOLDENS), default=512, help="the perplexity windows to score, in tokens")
    parser.add_argument("--command-timeout", type=int, default=COMMAND_TIMEOUT, metavar="SECONDS", help="the longest one llmx command may run, in seconds")
    parser.add_argument("--device", help="run the commands that take --device on this device or comma-separated list, e.g. vulkan:0")
    parser.add_argument("--file-exact", metavar="DIR", help="hold the file to the goldens tools/gen_baseline.py file-exact made from it in DIR, at 512-token windows, in place of its model's")
    args = parser.parse_args(argv)
    COMMAND_TIMEOUT = args.command_timeout
    if args.file_exact and args.context != 512:
        parser.error("file-exact goldens score 512-token windows")
    if args.device:
        os.environ["LLMX_DEVICE"] = args.device
    # The file is known by its name, which a link into the HF cache keeps and its target does not.
    exe, model, out = args.exe.resolve(), Path(os.path.abspath(args.model)), args.output_dir.resolve()
    if out.exists():
        parser.error("output directory already exists; use a new path")
    out.mkdir(parents=True)
    report = {"status": "running", "model": str(model), "executable": str(exe), "device": args.device or "cpu",
              "context": args.context, "commands": [], "checks": []}

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="ascii")

    def run(label, arguments):
        command = [str(exe)] + device_args(arguments)
        record = {"label": label, "command": command, "status": "running"}
        report["commands"].append(record)
        save()
        print(label, flush=True)
        try:
            result = subprocess.run(command, capture_output=True, timeout=COMMAND_TIMEOUT)
            stdout, stderr, rc = result.stdout, result.stderr, result.returncode
            record.update(status="exited", returncode=rc)
        except subprocess.TimeoutExpired as error:
            stdout, stderr, rc = error.stdout or b"", error.stderr or b"", None
            record.update(status="timeout", timeout_seconds=COMMAND_TIMEOUT)
        (out / (label + ".stdout")).write_bytes(stdout)
        (out / (label + ".stderr")).write_bytes(stderr)
        save()
        return rc, (stdout + stderr).decode("utf-8", "replace") if rc else stdout.decode("utf-8")

    def check(label, validator, *arguments):
        try:
            result = dict(status="pass", **validator(*arguments))
        except (ValueError, OverflowError) as error:
            result = dict(status="fail", error=str(error))
        report["checks"].append(dict(label=label, **result))
        save()
        return result["status"] == "pass"

    def write(name, data):
        (out / name).write_bytes(data)
        return str(out / name)

    save()
    try:
        report["executable_sha256"] = file_sha256(exe)
        directory = model.is_dir()
        if directory:
            require(not args.file_exact, "a directory is held to its model's goldens, which are file-exact for it")
            docs, report["model_sha256"] = directory_goldens(model)
            report["bounds"] = BOUNDS.get(FILE_EXACT_BOUNDS)
        else:
            report["model_sha256"] = file_sha256(model)
            report["bounds"] = model_goldens(str(model), args.file_exact)[1]
        rc, version = run("version", ["--version"])
        require(rc == 0, "llmx --version failed")
        report["version"] = version.strip()
        if directory:
            status = check_file(str(model), args.context, run, check, write, docs, report["bounds"])
        else:
            status = check_pinned(str(model), args.context, run, check, write, args.file_exact)
        failures = [item["label"] for item in report["checks"] if item["status"] == "fail"]
        require(not failures, "failed checks: " + ", ".join(failures))
        if status == "skip":
            report["status"] = "skip"
            print("qwen35 HF check SKIP - llmx does not run the qwen35 architecture here; %d tokenizer and chat id checks pass" % len(report["checks"]), flush=True)
        elif status == "unbounded":
            report["status"] = "unbounded"
            print("qwen35 HF check FAIL - %s has no bounds in tests/baseline_qwen35.py; %d checks measured, see report.json"
                  % (model.name, len(report["checks"])), flush=True)
        else:
            report["status"] = "pass"
            print("qwen35 HF check PASS: %d checks" % len(report["checks"]), flush=True)
    except (OSError, ValueError, OverflowError) as error:
        report.update(status="fail", error=str(error))
        print("qwen35 HF check FAIL: " + str(error), file=sys.stderr, flush=True)
    finally:
        save()
    return 0 if report["status"] in ("pass", "skip") else 1


if __name__ == "__main__":
    sys.exit(main())
