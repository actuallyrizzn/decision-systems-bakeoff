# clinc150 questions

Do **not** hardcode 151 criteria in git.

At run time, build the choice object from that task’s frozen `labels.json`
(alphabetical intent names, then `oos`):

```json
{
  "intent": {
    "type": "choice",
    "instructions": "Which of these intents does the utterance express?",
    "criteria": { "<label>": "<label>", "...": "...", "oos": "oos" }
  }
}
```

`scripts/run_arm.py` does this automatically when `--task clinc150`.
