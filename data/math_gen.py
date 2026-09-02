import random

def generate_math_problem(difficulty: int = 1):
    """
    Procedurally generates math problems with verifiable answers.
    Difficulty scales based on curriculum progress.
    """
    if difficulty == 1:
        # Basic arithmetic
        op = random.choice(['+', '-', '*'])
        a, b = random.randint(10, 100), random.randint(1, 50)
        
        if op == '+':
            ans = a + b
            text = f"What is {a} plus {b}?"
        elif op == '-':
            # Ensure positive answer for simplicity at level 1
            if a < b: a, b = b, a
            ans = a - b
            text = f"Subtract {b} from {a}."
        else:
            a, b = random.randint(5, 20), random.randint(5, 20)
            ans = a * b
            text = f"Multiply {a} by {b}."
            
        return text, str(ans)
        
    elif difficulty == 2:
        # Algebra: Solve for x
        x = random.randint(1, 20)
        a = random.randint(2, 10)
        b = random.randint(1, 50)
        c = a * x + b
        
        text = f"Solve for x: {a}x + {b} = {c}"
        return text, str(x)
        
    elif difficulty >= 3:
        # Word problems (Train speeds, mixtures, etc.)
        speed1 = random.randint(40, 80)
        time1_hrs = random.choice([1, 1.5, 2, 2.5])
        speed2 = random.randint(60, 100)
        time2_hrs = random.choice([1, 1.5, 2, 2.5])
        
        dist1 = speed1 * time1_hrs
        dist2 = speed2 * time2_hrs
        total_dist = dist1 + dist2
        total_time = time1_hrs + time2_hrs
        avg_speed = total_dist / total_time
        
        text = (f"A train travels at {speed1} km/h for {time1_hrs} hours, "
                f"then at {speed2} km/h for {time2_hrs} hours. "
                "What is the average speed for the entire journey in km/h?")
        
        return text, str(round(avg_speed, 2))


def generate_contrastive_problem():
    """
    Generates a contrastive reasoning problem for Phase 2 Error Correction.
    It deliberately introduces a common mistake, detects it, and corrects it.
    """
    x = random.randint(2, 10)
    a = random.randint(2, 5)
    b = random.randint(2, 15)
    c = a * x + b

    # Deliberate mistake: dividing before subtracting
    wrong_x = round((c / a) + b, 2)

    prompt = f"Solve: {a}x + {b} = {c}"
    response = (
        f"<problem>{prompt}</problem>\n"
        f"<wrong_attempt>x = {c}/{a} + {b} = {wrong_x} (INCORRECT)</wrong_attempt>\n"
        f"<error_detection>The error is: division was applied before subtracting {b} first.</error_detection>\n"
        f"<correct_thinking>{a}x = {c} - {b} = {c-b}, therefore x = {x}</correct_thinking>\n"
        f"<answer>{x}</answer>"
    )
    return prompt, response

def math_problem_stream(batch_size: int = 4, difficulty: int = 1, use_contrastive: bool = False):
    """Infinite generator of math problems."""
    while True:
        batch_prompts = []
        batch_answers = []
        for _ in range(batch_size):
            if use_contrastive and random.random() < 0.3:
                p, a = generate_contrastive_problem()
            else:
                p, a = generate_math_problem(difficulty)
            batch_prompts.append(p)
            batch_answers.append(a)
        yield batch_prompts, batch_answers
