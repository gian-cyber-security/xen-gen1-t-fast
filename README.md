# XEN-GEN1-T-Fast

Specialized XEN text model trained from scratch.

## Skills
Built-in skills in skills/*/SKILL.md are loaded at inference time. They are useful for users who do not want or cannot afford additional training. Training a specialized checkpoint can improve capability beyond skills alone.

## Train
~~~bash
python training/train.py --data datasets/train.jsonl --output outputs/xen --steps 1000
~~~

## Generate
~~~bash
python inference/generate.py --model-dir outputs/xen --prompt "your prompt here"
~~~

Use --no-skills to disable the skill layer.

Windows, Linux and macOS are supported.

## License
MIT
