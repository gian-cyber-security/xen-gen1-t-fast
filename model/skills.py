from pathlib import Path
def load_skills(root="skills"):
 p=Path(root); return [x.read_text(encoding="utf-8").strip() for x in sorted(p.glob("*/SKILL.md")) if x.is_file()] if p.exists() else []
def build_context(system,skills): return system.strip()+("\n\n--- XEN SKILLS ---\n"+"\n\n".join(skills) if skills else "")
