# Preflight report — v3

## dialogpt-small — BLOCKED

- chat templates not manually reviewed/confirmed

```
I'm convinced this feeling of dread will lead to relapse.<|endoftext|> That feeling of dread sounds intense, and the thought linking it directly to relapse is scary. Let's separate them. Notice the feeling of dread itself, where is it in your body? Now notice the *thought* about relapse. Is the thought definitely true, or is it a fearful prediction? What actions can you take right now to support your recovery *despite* the dread?
```

## gemma-3-1b-it — BLOCKED

- chat templates not manually reviewed/confirmed

```
<bos><start_of_turn>user
I'm convinced this feeling of dread will lead to relapse.<end_of_turn>
<start_of_turn>model
 That feeling of dread sounds intense, and the thought linking it directly to relapse is scary. Let's separate them. Notice the feeling of dread itself, where is it in your body? Now notice the *thought* about relapse. Is the thought definitely true, or is it a fearful prediction? What actions can you take right now to support your recovery *despite* the dread?
```

## gemma-4-e2b-it — BLOCKED

- chat template automated check failed: ["expected special token '<start_of_turn>' not found", "expected special token '<end_of_turn>' not found"]
- chat templates not manually reviewed/confirmed

```
<bos><|turn>user
I'm convinced this feeling of dread will lead to relapse.<turn|>
<|turn>model
 That feeling of dread sounds intense, and the thought linking it directly to relapse is scary. Let's separate them. Notice the feeling of dread itself, where is it in your body? Now notice the *thought* about relapse. Is the thought definitely true, or is it a fearful prediction? What actions can you take right now to support your recovery *despite* the dread?
```

## starling-lm-7b-alpha — BLOCKED

- chat templates not manually reviewed/confirmed

```
<s>GPT4 Correct User: I'm convinced this feeling of dread will lead to relapse.<|end_of_turn|>GPT4 Correct Assistant: That feeling of dread sounds intense, and the thought linking it directly to relapse is scary. Let's separate them. Notice the feeling of dread itself, where is it in your body? Now notice the *thought* about relapse. Is the thought definitely true, or is it a fearful prediction? What actions can you take right now to support your recovery *despite* the dread?
```

## empathetic-qwen3-8b-jan — BLOCKED

- chat templates not manually reviewed/confirmed

```
<|im_start|>user
I'm convinced this feeling of dread will lead to relapse.<|im_end|>
<|im_start|>assistant
<think>

</think>

 That feeling of dread sounds intense, and the thought linking it directly to relapse is scary. Let's separate them. Notice the feeling of dread itself, where is it in your body? Now notice the *thought* about relapse. Is the thought definitely true, or is it a fearful prediction? What actions can you take right now to support your recovery *despite* the dread?
```

## qwen3-8b — BLOCKED

- chat templates not manually reviewed/confirmed

```
<|im_start|>user
I'm convinced this feeling of dread will lead to relapse.<|im_end|>
<|im_start|>assistant
<think>

</think>

 That feeling of dread sounds intense, and the thought linking it directly to relapse is scary. Let's separate them. Notice the feeling of dread itself, where is it in your body? Now notice the *thought* about relapse. Is the thought definitely true, or is it a fearful prediction? What actions can you take right now to support your recovery *despite* the dread?
```

## mistral-7b-instruct-v0.3 — BLOCKED

- chat templates not manually reviewed/confirmed

```
<s>[INST] I'm convinced this feeling of dread will lead to relapse.[/INST] That feeling of dread sounds intense, and the thought linking it directly to relapse is scary. Let's separate them. Notice the feeling of dread itself, where is it in your body? Now notice the *thought* about relapse. Is the thought definitely true, or is it a fearful prediction? What actions can you take right now to support your recovery *despite* the dread?
```

## llama-3-8b — BLOCKED

- chat template automated check failed: ["expected special token '<|begin_of_text|>' not found", "expected special token '<|eot_id|>' not found"]
- chat templates not manually reviewed/confirmed

```
User: I'm convinced this feeling of dread will lead to relapse.
Bot: That feeling of dread sounds intense, and the thought linking it directly to relapse is scary. Let's separate them. Notice the feeling of dread itself, where is it in your body? Now notice the *thought* about relapse. Is the thought definitely true, or is it a fearful prediction? What actions can you take right now to support your recovery *despite* the dread?
```
