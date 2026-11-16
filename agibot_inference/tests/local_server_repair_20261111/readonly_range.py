import os,sys
p='/home/descfly/code/agibot_inference/model_servers/mid_a1_final/rootfs/mnt/workspace/users/haoran/workspace/runs/real_posttrain_c0_midfour_three_30k_n4_20261106_v1/A1/formal/train/checkpoint_step_30000.safetensors'
s=os.environ.get("SSH_ORIGINAL_COMMAND", "")
if not s.isdigit(): raise SystemExit("expected block index")
i=int(s)
if not 0<=i<12830: raise SystemExit("invalid block")
with open(p,"rb",buffering=0) as f:
 f.seek(i*(1<<20));raw=f.read(1<<20)
 sys.stdout.buffer.write(raw)
