
import random
import torch

def generate_sft_dataset(filename="/opt/rishabh_experimental/data/sft_math.pt", num_samples=10000):
    data = []
    for _ in range(num_samples):
        speed = random.randint(10, 120)
        time = round(random.uniform(0.5, 10.0), 1)
        distance = round(speed * time, 2)
        
        # The Golden Chain of Thought
        text = f"<problem>A train travels at {speed} km/h for {time} hours. How far did the train travel in total?</problem><thinking>To find the total distance, we use the formula: Distance = Speed * Time. We are given the speed is {speed} km/h and the time is {time} hours. Therefore, Distance = {speed} * {time} = {distance}.</thinking><answer>{distance}</answer>"
        
        # Encode
        tokens = [ord(c) % 32000 for c in text]
        data.append(torch.tensor(tokens, dtype=torch.long))
        
    torch.save(data, filename)
    print(f"Generated {num_samples} SFT examples at {filename}")

if __name__ == "__main__":
    import os
    os.makedirs("/opt/rishabh_experimental/data", exist_ok=True)
    generate_sft_dataset()
