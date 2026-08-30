#!/bin/bash
# Vigia da corrida G3: quando 27515 (gpu-node-a) OU 27521 (b200) entrar em R,
# cancela o outro, notifica e sai. Empate (ambos R) -> mantem b200, cancela gpu-node-a.
J_H=27515
J_B=27521
NTFY=https://ntfy.sh/daniels-code-a8f3k2608
for i in $(seq 1 5760); do   # ~48h a 30s
  sh=$(squeue -j $J_H -h -o "%t" 2>/dev/null)
  sb=$(squeue -j $J_B -h -o "%t" 2>/dev/null)
  if [ "$sh" = "R" ] && [ "$sb" = "R" ]; then
    scancel $J_H
    curl -s $NTFY -d "G3 corrida: AMBOS comecaram; mantive b200 (27521), cancelei gpu-node-a (27515)."
    echo "both R -> kept 27521, cancelled 27515"; exit 0
  fi
  if [ "$sh" = "R" ]; then
    scancel $J_B
    curl -s $NTFY -d "G3 corrida: gpu-node-a (27515, 4xH100) comecou; cancelei o b200 (27521). Slot QOS livre p/ Tasks B/C."
    echo "27515 R -> cancelled 27521"; exit 0
  fi
  if [ "$sb" = "R" ]; then
    scancel $J_H
    curl -s $NTFY -d "G3 corrida: b200 (27521, 2xB200) comecou; cancelei o gpu-node-a (27515). H100 livre p/ colega; slot QOS livre p/ Tasks B/C."
    echo "27521 R -> cancelled 27515"; exit 0
  fi
  if [ -z "$sh" ] && [ -z "$sb" ]; then
    curl -s $NTFY -d "G3 watcher: ambos os jobs sumiram da fila (cancelados/falharam?). Vigia encerrado."
    echo "both gone -> exit"; exit 0
  fi
  sleep 30
done
curl -s $NTFY -d "G3 watcher: 48h sem nenhum job comecar; vigia encerrado por timeout."
echo "timeout"
