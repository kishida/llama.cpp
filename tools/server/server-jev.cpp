#include "server-jev.h"

#include <algorithm>
#include <cmath>
#include <set>

json jev_error::to_json() const {
    json err = {
        {"code", code},
        {"message", what()},
    };
    if (!field.empty()) {
        err["field"] = field;
    }
    return json{{"error", err}};
}

// strings are used as-is, objects / arrays are serialized as JSON
static std::string jev_as_text(const json & v, bool pretty) {
    if (v.is_string()) {
        return v.get<std::string>();
    }
    return pretty ? v.dump(2) : v.dump();
}

static bool jev_is_structured(const json & v) {
    return (v.is_string() && v.get<std::string>().find_first_not_of(" \t\r\n") != std::string::npos) ||
           (v.is_object() && !v.empty()) || (v.is_array() && !v.empty());
}

jev_request jev_parse_request(const json & body, size_t max_options, const std::string & default_assistant_prefix) {
    if (!body.is_object()) {
        throw jev_error("invalid_request", "request body must be a JSON object");
    }
    jev_request req;
    req.assistant_prefix = default_assistant_prefix;

    if (!body.contains("state") || !jev_is_structured(body.at("state"))) {
        throw jev_error("invalid_state", "state must be a non-empty string, object, or array", "state");
    }
    req.state = jev_as_text(body.at("state"), true);

    if (body.contains("images")) {
        const json & imgs = body.at("images");
        if (!imgs.is_array()) {
            throw jev_error("invalid_images", "images must be an array of image URLs or base64 strings", "images");
        }
        for (size_t i = 0; i < imgs.size(); i++) {
            if (!imgs.at(i).is_string() || imgs.at(i).get<std::string>().empty()) {
                throw jev_error("invalid_images", "each image must be a non-empty string (data URL, base64, or http(s) URL)", "images." + std::to_string(i));
            }
            req.images.push_back(imgs.at(i).get<std::string>());
        }
    }

    if (!body.contains("model") || !body.at("model").is_string() || body.at("model").get<std::string>().empty()) {
        throw jev_error("missing_field", "model is required", "model");
    }

    if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
        throw jev_error("invalid_questions", "questions must be a non-empty map of question id to question", "questions");
    }

    for (const auto & item : body.at("questions").items()) {
        const std::string & id = item.key();
        const json & q = item.value();
        const std::string f = "questions." + id;
        if (!q.is_object()) {
            throw jev_error("invalid_question", "question must be an object", f);
        }
        if (!q.contains("instructions") || !jev_is_structured(q.at("instructions"))) {
            throw jev_error("invalid_instructions", "instructions must be a non-empty string, object, or array", f + ".instructions");
        }
        const std::string type = q.contains("type") && q.at("type").is_string() ? q.at("type").get<std::string>() : "";

        jev_question jq;
        jq.id           = id;
        jq.type         = type;
        jq.instructions = jev_as_text(q.at("instructions"), false);

        if (type == "noul") {
            jq.keys  = {"true", "false"};
            jq.texts = {"yes", "no"};
            if (q.contains("criteria")) {
                const json & c = q.at("criteria");
                if (!c.is_object()) {
                    throw jev_error("invalid_criteria", "noul criteria must be an object with optional true / false", f + ".criteria");
                }
                for (const auto & kv : c.items()) {
                    if (kv.key() != "true" && kv.key() != "false") {
                        throw jev_error("invalid_criteria", "unknown noul criteria key \"" + kv.key() + "\"", f + ".criteria." + kv.key());
                    }
                    if (!jev_is_structured(kv.value())) {
                        throw jev_error("invalid_criteria", "noul criteria values must be non-empty", f + ".criteria." + kv.key());
                    }
                    const size_t i = kv.key() == "true" ? 0 : 1;
                    jq.texts[i] += ": " + jev_as_text(kv.value(), false);
                }
            }
        } else if (type == "choice") {
            if (!q.contains("criteria") || !q.at("criteria").is_object()) {
                throw jev_error("invalid_criteria", "choice criteria must be a map of option to description (or null)", f + ".criteria");
            }
            for (const auto & kv : q.at("criteria").items()) {
                if (kv.key().find_first_not_of(" \t\r\n") == std::string::npos) {
                    throw jev_error("invalid_criteria", "option names must be non-empty", f + ".criteria");
                }
                if (!kv.value().is_null() && !jev_is_structured(kv.value())) {
                    throw jev_error("invalid_criteria", "option description must be a string, object, array, or null", f + ".criteria." + kv.key());
                }
                jq.keys.push_back(kv.key());
                jq.texts.push_back(kv.value().is_null() ? kv.key() : kv.key() + ": " + jev_as_text(kv.value(), false));
            }
        } else if (type == "score") {
            if (!q.contains("criteria") || !q.at("criteria").is_array()) {
                throw jev_error("invalid_criteria", "score criteria must be an ordered array of level descriptions", f + ".criteria");
            }
            const json & c = q.at("criteria");
            for (size_t i = 0; i < c.size(); i++) {
                if (!jev_is_structured(c.at(i))) {
                    throw jev_error("invalid_criteria", "level description must be a non-empty string, object, or array", f + ".criteria." + std::to_string(i));
                }
                jq.keys.push_back(std::to_string(i));
                jq.texts.push_back(jev_as_text(c.at(i), false));
                jq.legend.push_back(jev_as_text(c.at(i), false));
            }
        } else {
            throw jev_error("invalid_type", "type must be \"noul\", \"choice\", or \"score\"", f + ".type");
        }

        const std::string what = type == "score" ? "levels" : "options";
        if (jq.keys.size() < 2) {
            throw jev_error("too_few_" + what, type + " criteria must contain at least 2 " + what, f + ".criteria");
        }
        if (jq.keys.size() > max_options) {
            throw jev_error("too_many_" + what, "this model supports at most " + std::to_string(max_options) + " " + what + " per question", f + ".criteria");
        }
        req.questions.push_back(std::move(jq));
    }

    if (body.contains("options")) {
        const json & o = body.at("options");
        if (!o.is_object()) {
            throw jev_error("invalid_options", "options must be an object", "options");
        }
        if (o.contains("temperature_scaling")) {
            if (!o.at("temperature_scaling").is_boolean()) {
                throw jev_error("invalid_options", "options.temperature_scaling must be a boolean", "options.temperature_scaling");
            }
            req.temperature_scaling = o.at("temperature_scaling").get<bool>();
        }
        if (o.contains("temperature")) {
            const json & t = o.at("temperature");
            if (!t.is_number() || t.get<double>() <= 0) {
                throw jev_error("invalid_options", "options.temperature must be a positive number", "options.temperature");
            }
            req.temperature = (float) t.get<double>();
        }
        if (o.contains("return_logits")) {
            if (!o.at("return_logits").is_boolean()) {
                throw jev_error("invalid_options", "options.return_logits must be a boolean", "options.return_logits");
            }
            req.return_logits = o.at("return_logits").get<bool>();
        }
        req.assistant_prefix = jev_parse_assistant_prefix(body, default_assistant_prefix);
        if (o.contains("permutations")) {
            const json & p = o.at("permutations");
            if (!p.is_number_integer() || p.get<int>() < 1 || p.get<int>() > 64) {
                throw jev_error("invalid_options", "options.permutations must be an integer in [1, 64]", "options.permutations");
            }
            req.permutations = p.get<int>();
        }
    }
    return req;
}

std::string jev_parse_assistant_prefix(const json & body, const std::string & fallback) {
    if (!body.is_object() || !body.contains("options") || !body.at("options").is_object()) {
        return fallback;
    }
    const json & o = body.at("options");
    if (!o.contains("assistant_prefix")) {
        return fallback;
    }
    if (!o.at("assistant_prefix").is_string()) {
        throw jev_error("invalid_options", "options.assistant_prefix must be a string", "options.assistant_prefix");
    }
    return o.at("assistant_prefix").get<std::string>();
}

const std::vector<std::string> & jev_label_candidates() {
    static const std::vector<std::string> labels = [] {
        std::vector<std::string> v;
        for (char c = 'A'; c <= 'Z'; c++) v.emplace_back(1, c);
        for (char c = 'a'; c <= 'z'; c++) v.emplace_back(1, c);
        for (char c = '0'; c <= '9'; c++) v.emplace_back(1, c);
        return v;
    }();
    return labels;
}

std::string jev_user_message(const jev_request & req, const jev_question & q, const std::vector<jev_label> & labels,
                             const std::vector<int> & perm) {
    std::string opts;
    for (size_t pos = 0; pos < perm.size(); pos++) {
        if (pos > 0) opts += "\n";
        opts += labels[pos].text + ": " + q.texts[perm[pos]];
    }
    return "Context:\n" + req.state + "\n\n"
           "Answer the question with only the label of the best option (the character before the colon), nothing else.\n"
           "Question: " + q.instructions + "\nOptions:\n" + opts;
}

static double jev_round(double x) {
    return std::round(x * 1e4) / 1e4;
}

json jev_answer(const jev_question & q, const std::vector<std::vector<float>> & logits,
                const std::vector<std::vector<int>> & perms, float temperature, bool return_logits) {
    const size_t n = q.keys.size();
    std::vector<double> acc(n, 0.0); // probabilities in the original option order (averaged over permutations)
    std::vector<double> raw(n, 0.0); // raw label logits in the original option order (averaged over permutations)
    for (size_t k = 0; k < perms.size(); k++) {
        std::vector<double> z(n);
        double mx = -INFINITY;
        for (size_t pos = 0; pos < n; pos++) {
            raw[perms[k][pos]] += logits[k][pos] / perms.size();
            z[pos] = logits[k][pos] / temperature;
            mx = std::max(mx, z[pos]);
        }
        double sum = 0.0;
        for (auto & v : z) {
            v = std::exp(v - mx);
            sum += v;
        }
        for (size_t pos = 0; pos < n; pos++) {
            acc[perms[k][pos]] += z[pos] / sum / perms.size();
        }
    }

    // confidence = 1 - normalized entropy (0 for a uniform distribution, 1 when all mass is on one option)
    double h = 0.0;
    for (double p : acc) {
        if (p > 0) h -= p * std::log(p);
    }
    const double confidence = std::max(0.0, 1.0 - h / std::log((double) n));

    json ans;
    if (q.type == "noul") {
        ans = json{{"type", "noul"}, {"noul", jev_round(acc[0])}};
    } else {
        json probs = json::object();
        size_t best = 0;
        for (size_t i = 0; i < n; i++) {
            probs[q.keys[i]] = jev_round(acc[i]);
            if (acc[i] > acc[best]) best = i;
        }
        if (q.type == "choice") {
            ans = json{{"type", "choice"}, {"choice", q.keys[best]}, {"probabilities", probs}, {"confidence", jev_round(confidence)}};
        } else {
            double score = 0.0;
            json legend = json::object();
            for (size_t i = 0; i < n; i++) {
                score += i * acc[i];
                legend[q.keys[i]] = q.legend[i];
            }
            ans = json{{"type", "score"}, {"score", jev_round(score)}, {"legend", legend}, {"probabilities", probs}, {"confidence", jev_round(confidence)}};
        }
    }
    if (return_logits) {
        json lg = json::object();
        for (size_t i = 0; i < n; i++) {
            lg[q.keys[i]] = raw[i];
        }
        ans["logits"] = lg;
    }
    return ans;
}
