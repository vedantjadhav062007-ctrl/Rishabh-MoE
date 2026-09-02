import re

def extract_answer(text: str):
    """Extracts the final answer from the <answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def verify_math_answer(generated_answer: str, true_answer: str) -> bool:
    """
    Verifies if the generated numerical answer matches the true answer.
    Provides the reward signal for GRPO RL training.
    """
    if generated_answer is None or true_answer is None:
        return False
        
    # Clean up strings
    gen_clean = re.sub(r'[^\d\.\-]', '', generated_answer)
    true_clean = re.sub(r'[^\d\.\-]', '', true_answer)
    
    if not gen_clean or not true_clean:
        return False
        
    try:
        # Check numerical equivalence with a small epsilon
        return abs(float(gen_clean) - float(true_clean)) < 1e-6
    except ValueError:
        # If parsing fails, fall back to exact string match
        return generated_answer.strip().lower() == true_answer.strip().lower()

def compute_grpo_reward(output_text: str, true_answer: str) -> float:
    """
    Computes a simple GRPO reward.
    +1.0 for correct answer in <answer> tag
    +0.1 for correct formatting (having <thinking> and <answer> tags)
    -0.5 for missing answer tag
    -1.0 for incorrect answer
    """
    reward = 0.0
    
    has_thinking = "<thinking>" in output_text and "</thinking>" in output_text
    has_answer = "<answer>" in output_text and "</answer>" in output_text
    
    if has_thinking and has_answer:
        reward += 0.1
    elif not has_answer:
        reward -= 0.5
        
    extracted = extract_answer(output_text)
    if extracted:
        is_correct = verify_math_answer(extracted, true_answer)
        if is_correct:
            reward += 1.0
        else:
            reward -= 1.0
            
    return reward