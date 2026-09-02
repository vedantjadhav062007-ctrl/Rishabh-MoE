#!/bin/bash
while true; do
  cd /opt/rishabh_experimental
  # List files, keep top 2, delete the rest
  ls -t ckpt_step_*.pt 2>/dev/null | tail -n +3 | xargs -I {} rm -f -- {}
  sleep 600
done

