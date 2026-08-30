# --- guard: o Slurm ja nos deu GPU fisicamente ocupada por outro processo
# (3 falhas de OOM no sglang em 21/07). Falha rapido e re-enfileira em vez de
# queimar 6min e morrer silenciosamente. Requer #SBATCH --requeue.
gpu_guard() {
  local busy=0 idx used
  IFS=',' read -ra _GG <<< "${CUDA_VISIBLE_DEVICES}"
  for idx in "${_GG[@]}"; do
    used=$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
    echo "guard: GPU $idx com ${used} MiB em uso"
    if [ "$used" -gt 5000 ]; then busy=1; fi
  done
  if [ "$busy" = 1 ]; then
    if [ "${SLURM_RESTART_COUNT:-0}" -lt 10 ]; then
      echo "guard: GPU alocada ocupada por terceiros -> requeue em 5min (tentativa ${SLURM_RESTART_COUNT:-0}/10)"
      sleep 300
      scontrol requeue "$SLURM_JOB_ID"
      exit 0
    fi
    echo "guard: 10 requeues sem GPU limpa — desistindo."; exit 1
  fi
  echo "guard: GPUs limpas, seguindo."
}
gpu_guard
