
import os
import glob
import time

while True:
    try:
        # Find all ckpt_step_*.pt files in the experimental folder
        ckpts = glob.glob("/opt/rishabh_experimental/ckpt_step_*.pt")
        # Sort by modification time (oldest first)
        ckpts.sort(key=os.path.getmtime)
        
        # Keep the newest 2
        if len(ckpts) > 2:
            to_delete = ckpts[:-2]
            for ckpt in to_delete:
                os.remove(ckpt)
                print(f"Deleted old checkpoint: {ckpt}")
    except Exception as e:
        pass
    
    # Sleep for 10 minutes
    time.sleep(600)
