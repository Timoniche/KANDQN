memory_values=(
#  1
  128
  500
  1000
  2000
  5000
  10000
  20000
  50000
  100000
)

for memory in "${memory_values[@]}"; do
    echo "Running with memory = $memory"

    python ./../main.py --config_file "./../configs/efficient_vs_dqn/efficient_16_memory_${memory}.yaml" --wandb_enabled False
done

echo "All runs completed!"