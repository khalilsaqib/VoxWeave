from pathlib import Path

class PromptOptions:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]
        prompts_dir = project_root / "prompts"
        
        self.voice_rules_prompt = (
            prompts_dir / "voice_rules_prompt.txt"
        ).read_text(encoding="utf-8")

prompt_options = PromptOptions()
