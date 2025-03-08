memory_values=(
#  128
#  500
#  1000
  2000
  5000
  10000
  20000
  50000
  100000
)

for memory in "${memory_values[@]}"; do
    echo "Running with memory = $memory"

    python ./../main.py --config_file "./../configs/fkaqn_vs_dqn/fkaqn_16_memory_${memory}.yaml" --wandb_enabled False
    python ./../main.py --config_file "./../configs/fkaqn_vs_dqn/dqn_32_memory_${memory}.yaml" --wandb_enabled False
done

echo "All runs completed!"