import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json

def extract_tensorboard_data(tb_path):
    """Extract data from TensorBoard logs."""
    print(f"Parsing TensorBoard logs from: {tb_path}")
    
    # Get all event files
    event_files = glob.glob(os.path.join(tb_path, "events.out.tfevents.*"))
    if not event_files:
        print("No event files found")
        return {}
    
    # Use the first event file
    event_file = event_files[0]
    print(f"Loading: {os.path.basename(event_file)}")
    
    # Load TensorBoard data
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Extract scalar data
    data = {}
    print("\nAvailable metrics:")
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
        
        # Show some stats
        if len(values) > 0:
            print(f"  {tag}: {len(values)} points, last value: {values[-1]:.3f}")
    
    return data

def create_learning_curves(tb_data, output_dir="./results"):
    """Create learning curve plots from TensorBoard data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Main Learning Curve - Training Returns
    plt.figure(figsize=(12, 8))
    
    # Plot training returns
    if 'training/return' in tb_data:
        data = tb_data['training/return']
        plt.plot(data['steps'], data['values'], 'b-', linewidth=2, label='Training Return')
        
        # Add moving average for smoother curve
        if len(data['values']) > 10:
            window = max(1, len(data['values']) // 50)
            moving_avg = np.convolve(data['values'], np.ones(window)/window, mode='valid')
            plt.plot(data['steps'][window-1:], moving_avg, 'b--', alpha=0.7, 
                     label=f'Moving Avg (window={window})')
    
    # Plot evaluation returns
    if 'evaluation/average_returns' in tb_data:
        data = tb_data['evaluation/average_returns']
        plt.plot(data['steps'], data['values'], 'g-', linewidth=2.5, label='Evaluation Return', marker='o', markersize=4)
    
    plt.xlabel('Environment Steps')
    plt.ylabel('Return')
    plt.title('RL Learning Curve - CPC Gripper')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curve_xirllong.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "learning_curve_xirllong.pdf"))
    plt.show()
    
    # 2. Success Rate Plot (if eval_score exists)
    if 'evaluation/average_eval_scores' in tb_data:
        plt.figure(figsize=(10, 6))
        data = tb_data['evaluation/average_eval_scores']
        plt.plot(data['steps'], data['values'], 'r-', linewidth=2.5, marker='s', markersize=5)
        
        # Add horizontal line at paper's XIRL performance
        plt.axhline(y=0.28, color='green', linestyle='--', alpha=0.5, label='Paper XIRL (0.28)')
        
        
        # Mark final performance
        if len(data['values']) > 0:
            final_score = data['values'][-1]
            plt.axhline(y=final_score, color='blue', linestyle=':', alpha=0.7, 
                       label=f'Cpc ({final_score:.3f})')
        
        plt.xlabel('Environment Steps')
        plt.ylabel('Success Rate / Eval Score')
        plt.title('Success Rate Over Time')
        plt.ylim([0, 1.1])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "success_rate_xirllong.png"), dpi=150)
        plt.savefig(os.path.join(output_dir, "success_rate_xirllong.pdf"))
        plt.show()
    
    # 3. Loss Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Actor loss
    if 'training/actor_loss' in tb_data:
        ax = axes[0, 0]
        data = tb_data['training/actor_loss']
        ax.plot(data['steps'], data['values'], 'purple', linewidth=1.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss')
        ax.grid(True, alpha=0.3)
    
    # Critic loss
    if 'training/critic_loss' in tb_data:
        ax = axes[0, 1]
        data = tb_data['training/critic_loss']
        ax.plot(data['steps'], data['values'], 'orange', linewidth=1.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Critic Loss')
        ax.set_title('Critic Loss')
        ax.grid(True, alpha=0.3)
    
    # Episode length
    if 'training/length' in tb_data:
        ax = axes[1, 0]
        data = tb_data['training/length']
        ax.plot(data['steps'], data['values'], 'teal', linewidth=1.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
    
    # Temperature
    if 'training/temperature' in tb_data:
        ax = axes[1, 1]
        data = tb_data['training/temperature']
        ax.plot(data['steps'], data['values'], 'brown', linewidth=1.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Temperature')
        ax.set_title('SAC Temperature')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_details_xirllong.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "training_details_cpcxirllong.pdf"))
    plt.show()
    
    # 4. Final Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    
    # Paper values from XIRL paper (Table 1 - adjust based on actual paper)
    methods = ['My Xirl at 100k steps', 'Paper XIRL', 'TCN', 'LIFS', 'Goal Classifier']
    
    # Get your final eval score
    your_score = 0.0
    if 'evaluation/average_eval_scores' in tb_data:
        data = tb_data['evaluation/average_eval_scores']
        if len(data['values']) > 0:
            your_score = data['values'][-1]
    
    # Paper values (example - replace with actual from paper)
    paper_scores = [your_score, 0.75, 0.35, 0.13, 0.14]
    
    colors = ['blue', 'red', 'gray', 'gray', 'gray']
    bars = plt.bar(methods, paper_scores, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, score in zip(bars, paper_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Success Rate')
    plt.title('Final Performance Comparison')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_comparison_xirlmedium.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "final_comparison_xirlmedium.pdf"))
    plt.show()
    
    # 5. Save data for later analysis
    save_data = {}
    for key, value in tb_data.items():
        save_data[key] = {
            'steps': value['steps'],
            'values': value['values']
        }
    
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nAll visualizations saved to: {output_dir}/")

def main():
    print("=== CPC Results Visualization ===")
    print("Creating plots from completed training run...")
    
    # Define the run directory (from your output)
    run_dir = r"/tmp/xirl/rl_runs/env_name=SweepToTop-Gripper-State-Allo-TestLayout-v0_reward=learned_reward_type=distance_to_goal_mode=same_algo=xirl_uid=4f5ae804-248b-4fb9-8d3a-bed322e579fe/0"
    
    if not os.path.exists(run_dir):
        print(f"ERROR: Run directory not found: {run_dir}")
        return
    
    print(f"Run directory: {run_dir}")
    
    # Extract TensorBoard data
    tb_path = os.path.join(run_dir, "tb")
    if not os.path.exists(tb_path):
        print(f"ERROR: TensorBoard directory not found: {tb_path}")
        return
    
    tb_data = extract_tensorboard_data(tb_path)
    
    if not tb_data:
        print("ERROR: No TensorBoard data extracted")
        return
    
    # Create visualizations
    create_learning_curves(tb_data)
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    if 'evaluation/average_eval_scores' in tb_data:
        eval_data = tb_data['evaluation/average_eval_scores']
        if len(eval_data['values']) > 0:
            print(f"Final Evaluation Score: {eval_data['values'][-1]:.3f}")
            print(f"Best Evaluation Score: {max(eval_data['values']):.3f}")
    
    if 'training/return' in tb_data:
        train_data = tb_data['training/return']
        if len(train_data['values']) > 0:
            print(f"Final Training Return: {train_data['values'][-1]:.3f}")
            print(f"Best Training Return: {max(train_data['values']):.3f}")
    
    print("\nNext: Compare your final score with paper results!")

if __name__ == "__main__":
    main()