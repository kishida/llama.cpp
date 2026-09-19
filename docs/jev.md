# Jev: classification with calibrated probabilities

`llama-server` can answer multiple-choice questions about a piece of text (or an image) and return a
**probability for every option**, instead of generating an answer as text.

How it works: each question becomes a prompt whose options are labelled `A`, `B`, `C`, ... The prompt is
evaluated once, and the probabilities are read directly from the logits of the next token. Nothing is
generated, so an answer costs one prompt evaluation and the result is a proper probability distribution you
can threshold, rank, or route on.

It works with **any model** — no fine-tuning needed. The API is compatible with
[TypeSafe System One](https://docs.typesafe.ai/api).

## Quick start

Start a server as usual:

```bash
llama-server -m model.gguf -ngl 99 --port 8080
```

Then open **http://localhost:8080/jev** in a browser for the demo page, or call the API:

```bash
curl http://localhost:8080/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "jev-latest",
  "state": "I have been trying to connect my Stripe account for 3 days and it keeps failing. I am losing sales.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this",
      "criteria": {
        "billing": "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales": "Pricing or account questions"
      }
    }
  }
}'
```

```json
{
  "model": "model.gguf",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.5783, "technical": 0.4083, "sales": 0.0134 },
      "confidence": 0.3261
    }
  },
  "usage": { "input_tokens": 98, "output_tokens": 0 }
}
```

`state` is the text every question is asked about; objects and arrays are serialized to JSON for you.
One request can carry **many questions about the same state** — they share the prompt prefix, so the prompt
cache handles the state only once.

## Question types

| type | ask for | answer fields |
|---|---|---|
| `choice` | one option out of several | `choice`, `probabilities`, `confidence` |
| `noul` | a yes/no judgement | `noul` (probability of true) |
| `score` | a level on an ordered scale | `score` (expected level), `legend`, `probabilities`, `confidence` |

`confidence` is `1 - normalized entropy`: 1 when all the mass is on one option, 0 when the options are
indistinguishable. `score` is the average level weighted by its probability, so 1.6 means "between level 1
and level 2, closer to 2".

```bash
curl http://localhost:8080/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "jev-latest",
  "state": "Third time I am writing. My order still has not arrived. Refund me.",
  "questions": {
    "is_angry":  { "type": "noul",  "instructions": "Is this customer angry?" },
    "urgency":   { "type": "score", "instructions": "How urgent is this inquiry?",
                   "criteria": ["Not urgent", "Within a few days", "Today", "Immediately"] }
  }
}'
```

```json
{
  "answers": {
    "is_angry": { "type": "noul", "noul": 0.9009 },
    "urgency": {
      "type": "score",
      "score": 2.0111,
      "legend": { "0": "Not urgent", "1": "Within a few days", "2": "Today", "3": "Immediately" },
      "probabilities": { "0": 0.2506, "1": 0.0707, "2": 0.0956, "3": 0.5831 },
      "confidence": 0.2259
    }
  },
  "usage": { "input_tokens": 159, "output_tokens": 0 }
}
```

`criteria` is optional for `noul` (`{"true": "...", "false": "..."}` to spell out what each side means),
a map of option name to description (or `null`) for `choice`, and an ordered array from lowest to highest
for `score`.

Invalid requests return HTTP 422 and name the offending field:

```json
{ "error": { "code": "invalid_criteria", "message": "...", "field": "questions.urgency.criteria" } }
```

## Images

Start the server with a multimodal projector, and images may accompany the state. They are shown to the
model before every question in the request.

```bash
llama-server -m model.gguf --mmproj mmproj.gguf -ngl 99 --port 8080
```

```bash
curl http://localhost:8080/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "jev-latest",
  "state": "Answer by looking at the photo.",
  "images": ["data:image/jpeg;base64,/9j/4AAQSkZJRg..."],
  "questions": {
    "scene": { "type": "choice", "instructions": "What kind of scene is this?",
               "criteria": { "meal": null, "work": null, "sports": null, "travel": null, "other": null } }
  }
}'
```

Each entry of `images` is a `data:` URL, a bare base64 string, or an `http(s)://` URL. A server started
without `--mmproj` answers with 422 `images_not_supported`.

## Calibration

The probabilities are only useful if they are honest: when the model says 0.8, it should be right about 80%
of the time. Most models are over-confident, and the usual fix is one number — a temperature that the logits
are divided by.

Pass it per request:

```json
"options": { "temperature": 3.6 }
```

To find that number, collect the raw logits on a few hundred labelled examples with
`"options": { "return_logits": true }` and fit the temperature that minimizes the negative log-likelihood.
Typical values: about 3.6 for Gemma 4 12B, about 0.84 for Qwen3.5 2B — both bring the expected calibration
error to roughly 0.03.

A model fine-tuned for this task (GGUF metadata key `jev.temperature`) carries its own temperature and uses
the fixed prompt format it was trained with; you do not need to pass anything.

Other options:

| option | default | meaning |
|---|---|---|
| `temperature` | model's own, else 1 | divide the logits by this |
| `temperature_scaling` | `true` | set to `false` to ignore the temperature entirely |
| `permutations` | 1 | average over K rotations of the option order, to cancel position bias (costs K evaluations) |
| `return_logits` | `false` | also return the raw label logits, for fitting a temperature |

## Web UI

`GET /jev` serves a small demo page: enter a state, add questions, and see the probabilities as bars next to
the request and response JSON. It picks up the loaded model from `/props` and offers an image field when the
server was started with `--mmproj`. It is handy for trying prompts and options before wiring the API into
anything — and for reading the request JSON it builds, which you can paste straight into `curl`.

## Notes and limits

- Options are labelled with single-token symbols (`A`-`Z`, `a`-`z`, `0`-`9`), so a question can have at most
  as many options as the model has such single-token symbols — usually 62.
- Longer option descriptions generally help; the label is just a handle.
- `model` in the request is required by the API but ignored, except in router mode where it selects the model.
- Nothing is generated, so `output_tokens` is always 0 and sampling parameters do not apply.
