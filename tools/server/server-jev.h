#pragma once

// Jev (TypeSafe System One) compatible classification endpoint: POST /v1/systemone
// https://docs.typesafe.ai/api
//
// Extension: "images": ["data:image/...;base64,...", "https://...", ...] adds images to the state (requires a
// multimodal model loaded with --mmproj). The images are placed at the start of every question's prompt.
//
// Every question is turned into a multiple-choice prompt whose options are labelled with single-token
// symbols (A-Z, a-z, 0-9). The prompt is evaluated once and the probability of each option is read from the
// logits of the next token (no text is generated). Questions that share the same state share the prompt prefix,
// so the prompt cache can reuse it.
//
// The prompt is built with the model's own chat template, with thinking disabled so that the answer label is the
// next token. Models are usually over-confident, so pass a calibrated temperature with the extension
// "options": {"temperature": T} (fit it on labelled data using "options": {"return_logits": true}). A model
// fine-tuned for this task can ship its own temperature in the GGUF metadata key "jev.temperature".

#include "server-common.h"

#include <stdexcept>
#include <string>
#include <vector>

// validation error, returned as HTTP 422 {"error": {"code", "message", "field"}}
struct jev_error : std::runtime_error {
    std::string code;
    std::string field;
    jev_error(const std::string & code, const std::string & message, const std::string & field = "")
        : std::runtime_error(message), code(code), field(field) {}
    json to_json() const;
};

struct jev_question {
    std::string id;
    std::string type;                // "noul" | "choice" | "score"
    std::string instructions;        // question text put into the prompt
    std::vector<std::string> keys;   // answer keys: choice option names / score levels "0".."n-1" / noul "true","false"
    std::vector<std::string> texts;  // option texts put into the prompt
    std::vector<std::string> legend; // score only: level descriptions
};

struct jev_request {
    std::string state;               // state as text (objects / arrays are serialized as JSON)
    std::vector<std::string> images; // extension: images shown with the state (data URL / base64 / http(s) URL)
    std::vector<jev_question> questions;
    bool  temperature_scaling = true;  // extension: "options": {"temperature_scaling": false}
    float temperature         = 0.0f;  // extension: "options": {"temperature": T} overrides the model's temperature (0 = not set)
    int   permutations        = 1;     // extension: "options": {"permutations": K} averages K cyclic option orders
    bool  return_logits       = false; // extension: "options": {"return_logits": true} adds the raw label logits to each answer
};

// a label symbol and the token the model produces for it right after the generation prompt
struct jev_label {
    std::string text;
    llama_token token;
};

// parse and validate the request body (throws jev_error)
jev_request jev_parse_request(const json & body, size_t max_options);

// candidate label symbols in the order they are assigned: A-Z, a-z, 0-9
const std::vector<std::string> & jev_label_candidates();

// user message for a question, with options presented in the order given by perm (perm[k] = original option index)
std::string jev_user_message(const jev_request & req, const jev_question & q, const std::vector<jev_label> & labels,
                             const std::vector<int> & perm);


// build the answer from the label logits of each permutation (logits[k][pos] for perms[k])
json jev_answer(const jev_question & q, const std::vector<std::vector<float>> & logits,
                const std::vector<std::vector<int>> & perms, float temperature, bool return_logits);
