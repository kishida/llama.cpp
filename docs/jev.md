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

This endpoint is not in upstream llama.cpp yet, so build the `jev` branch of the fork:

```bash
git clone -b jev https://github.com/kishida/llama.cpp
cd llama.cpp
```

Windows with an NVIDIA GPU (needs the CUDA toolkit and Visual Studio):

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --target llama-server -j 16
```

macOS (Metal is on by default, nothing to configure):

```bash
cmake -B build
cmake --build build --config Release --target llama-server -j 8
```

Pass `-j` the number of cores you want to build with. The binary lands in `build/bin/Release/llama-server.exe`
on Windows and `build/bin/llama-server` on macOS. See [build.md](build.md) for other backends.

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

Encoding an image costs far more than answering a question about it, so all the questions of one request are
run on a single slot, one after the other, and the work up to the end of the shared prefix — the images and
the state — is done once and reused. Ask everything you want about a picture in one request: on Qwen3.5 2B,
eight questions about one photo take 1.6 s instead of 4.6 s.

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

A model fine-tuned for this task can ship its temperature in the GGUF metadata key `jev.temperature`; it is
then used by default and you do not need to pass anything.

Other options:

| option | default | meaning |
|---|---|---|
| `temperature` | model's own, else 1 | divide the logits by this |
| `temperature_scaling` | `true` | set to `false` to ignore the temperature entirely |
| `permutations` | 1 | average over K rotations of the option order, to cancel position bias (costs K evaluations) |
| `return_logits` | `false` | also return the raw label logits, for fitting a temperature |
| `assistant_prefix` | detected from the chat template | text appended after the generation prompt, to start the assistant turn (see below) |

## Prompt format

Each question becomes one user message, built like this and then wrapped in the model's own chat template
(with an assistant generation prompt, thinking disabled):

```
Context:
<state>

Answer the question with only the label of the best option (the character before the colon), nothing else.
Question: <instructions>
Options:
A: <first option>
B: <second option>
```

The options are written as `name: description`, or just `name` when the description is `null`. A `noul`
question becomes `yes` / `no` (plus your `criteria` text if you gave any), and a `score` question becomes its
levels in order, lowest first. The labels `A`, `B`, ... are assigned per question, and only symbols that are a
single token right after the generation prompt are used, so the answer is always exactly one token.

That token is where the probabilities come from: the server evaluates the prompt once and reads the logits of
the label tokens at that position.

**If you fine-tune a model for this**, train it on this same format — the same wording, the same option
layout, and one label character as the target — and the server will use it as is. You can also store the
temperature you fitted in the GGUF metadata key `jev.temperature` so clients get calibrated probabilities
without passing anything.

## Web UI

`GET /jev` serves a small demo page: enter a state, add questions, and see the probabilities as bars next to
the request and response JSON. It picks up the loaded model from `/props` and offers an image field when the
server was started with `--mmproj`. It is handy for trying prompts and options before wiring the API into
anything — and for reading the request JSON it builds, which you can paste straight into `curl`.

## When the label is not the next token

Some models do not start their answer with the answer, and then the token after the generation prompt is not a
label and the probabilities are meaningless. The server catches the common case by itself: it diffs the chat
template rendered with an assistant message against the generation prompt, which finds the channel marker of a
harmony-style model (gpt-oss, LLM-jp-4). Without it those two score like guessing; with it they are among the
best models here. A line in the log says when one was found:

```
jev: assistant prefix detected from the chat template: "<|channel|>final<|message|>"
```

Detection only sees what the template writes. A model that opens a reasoning block on its own — LFM2.5 8B
starts with `<think>` although its template does not — needs to be told, which also takes its accuracy on our
benchmark from 0.50 to 0.59:

```bash
llama-server -m LFM2.5-8B-A1B.gguf -ngl 99 --jev-assistant-prefix "<think>\n\n</think>\n\n"
```

The text is appended after the generation prompt, with `\n` and the other usual escapes expanded. A request
can override it per question set with
`"options": {"assistant_prefix": "..."}`, and an empty string turns the detection off.

## Notes and limits

- Options are labelled with single-token symbols (`A`-`Z`, `a`-`z`, `0`-`9`), so a question can have at most
  as many options as the model has such single-token symbols — usually 62.
- Longer option descriptions generally help; the label is just a handle.
- `model` in the request is required by the API but ignored, except in router mode where it selects the model.
- Nothing is generated, so `output_tokens` is always 0 and sampling parameters do not apply.

## Reference: which models are worth using

Numbers from one benchmark — 1,191 multiple-choice questions with 2 to 8 options, where always guessing gives
0.283 — so read them as a rough ordering, not as a score for the model. The temperature was fitted on a
separate 358-question split, and ECE is shown before and after applying it.

| model | accuracy | ECE (T=1 → calibrated) | T | notes |
|---|---|---|---|---|
| Gemma 4 12B UD-Q4_K_XL | 0.897 | 0.092 → 0.030 | 3.60 | |
| gpt-oss 20B MXFP4 | 0.840 | 0.059 → 0.025 | 1.53 | assistant prefix, detected |
| LLM-jp-4 8B thinking Q4_K_M | 0.825 | 0.099 → 0.041 | 1.69 | assistant prefix, detected |
| Qwen3.5 2B Q8_0 | 0.711 | 0.040 → 0.030 | 0.84 | calibrated as it comes |
| Qwen3 1.7B Q8_0 | 0.712 | 0.270 → 0.038 | 8.56 | |
| LFM2.5 8B A1B UD-Q4_K_XL | 0.591 | 0.246 → 0.048 | 2.70 | needs `--jev-assistant-prefix`; 0.496 without it |
| LFM2.5 350M Q8_0 | 0.572 | 0.347 → 0.055 | 6.70 | |
| gemma-3 270m-it Q8_0 | 0.287 | 0.341 → 0.032 | 13.44 | no better than guessing |
| (guessing) | 0.283 | | | |

What it suggests:

- A small instruct model is enough to be useful, but not any small model: at 270M the answers are noise, and
  a large temperature then only makes the model uniformly unsure rather than right.
- Accuracy and calibration are separate problems. Qwen3.5 2B is honest out of the box (T = 0.84) while Qwen3
  1.7B answers almost everything with near-certainty until it is divided by 8.6. Fit the temperature.
- Check the assistant prefix before judging a model. gpt-oss and LLM-jp-4 look like random guessing without
  one, and LFM2.5 8B needs a prefix the server cannot detect, because the model opens `<think>` on its own
  rather than the template writing it.
- Mixture-of-experts models are priced by their active parameters here too: LFM2.5 8B A1B activates about 1B
  and scores like a small model.
