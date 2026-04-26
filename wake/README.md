# Custom wake words

Drop trained `.onnx` files in this directory and point `wake.model` in
`config.yaml` at the path:

```yaml
wake:
  model: wake/hey_dave.onnx
```

If the path doesn't exist, the assistant falls back to the bundled
`hey_jarvis` model and prints a warning.

## Training "hey dave"

openWakeWord ships an automated training pipeline that uses synthetic TTS
samples. The fast path is the official Colab notebook:

<https://github.com/dscripka/openWakeWord#training-new-models>

Rough steps:

1. Open `automatic_model_training.ipynb` in Colab (free GPU is enough).
2. Set the wake-word phrase to `hey dave`.
3. Run all cells. It generates ~30k synthetic positive samples with
   different voices, mixes them with negatives, and trains a small CNN.
   Training takes ~30-60 min on a Colab T4.
4. Download the resulting `hey_dave.onnx` and copy it here:

   ```bash
   scp hey_dave.onnx server@server:~/local-assistant/wake/
   ```

5. Edit `config.yaml`:

   ```yaml
   wake:
     model: wake/hey_dave.onnx
     threshold: 0.5
   ```

6. Run `./run.sh -v` and say "hey Dave" — you should see the
   `wake_score_peak` jump above the threshold and trigger the recorder.

## Tuning

If the model is too eager (false-fires) or too quiet (misses you):

- Raise `wake.threshold` to 0.6-0.7 to reduce false fires.
- Lower it to 0.3-0.4 to catch quieter speech.
- The cooldown (`wake.cooldown_sec`) prevents re-fire while the assistant
  is still speaking; bump it if the model is interrupting itself.

## Why not Picovoice / Porcupine

It works and is faster to set up, but requires a Picovoice access key
and a phone-home check on launch. openWakeWord stays fully local with no
account.
