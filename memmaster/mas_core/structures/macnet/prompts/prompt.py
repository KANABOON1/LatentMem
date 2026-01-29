def get_domain_prompt(task_domain: str):
    if task_domain == "kodcode":
        from memmaster.mas_core.structures.macnet.prompts import kodcode_prompt
        return kodcode_prompt
    elif task_domain == "triviaqa":
        from memmaster.mas_core.structures.macnet.prompts import triviaqa_prompt
        return triviaqa_prompt
    elif task_domain == "popqa":
        from memmaster.mas_core.structures.macnet.prompts import popqa_prompt
        return popqa_prompt
    elif task_domain == "pddl":
        from memmaster.mas_core.structures.macnet.prompts import pddl_prompt
        return pddl_prompt
    else:
        raise ValueError(f"Unsupported task domina: {task_domain}")
    