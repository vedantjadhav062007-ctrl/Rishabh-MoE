import torch
import traceback

print('PyTorch CUDA available:', torch.cuda.is_available())
try:
    print('Loading /opt/rishabh/rishabh_final.pt on CPU...')
    ckpt = torch.load('/opt/rishabh/rishabh_final.pt', map_location='cpu')
    print('Checkpoint Loaded Successfully!')
    print('Step:', ckpt.get('step'))
    print('Difficulty:', ckpt.get('difficulty'))
    print('Keys in ckpt:', list(ckpt.keys()))
    print('Optimizer master count:', len(ckpt.get('optimizer_master', [])))
except Exception as e:
    traceback.print_exc()
