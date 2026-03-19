import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

def find_rl_runs():
    """Find RL training directories."""
    # Try different possible locations
    check_paths = [
        r"C:\tmp\xirl\rl_runs",
        r"/tmp/xirl/rl_runs",
    ]
    
    for path in check_paths:
        if os.path.exists(path):
            print(f"Found directory: {path}")
            # List all experiments
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print(f"  Experiment: {item}")
                    # List seeds
                    for seed_item in os.listdir(item_path):
                        seed_path = os.path.join(item_path, seed_item)
                        if os.path.isdir(seed_path):
                            print(f"    Seed: {seed_item} -> {seed_path}")
                            return seed_path
    return None

def check_training_progress():
    """Check current training status."""
    print("=== XIRL Training Status ===")
    
    # Find runs
    run_dir = find_rl_runs()
    
    if not run_dir:
        print("No RL runs directory found yet.")
        print("This is normal if:")
        print("1. Training just started")
        print("2. No checkpoints/logs saved yet")
        print("3. Different save directory is being used")
        return
    
    print(f"\nAnalyzing run: {run_dir}")
    
    # Check what's in the directory
    print("\nDirectory contents:")
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path):
            print(f"  [DIR]  {item}/")
        else:
            size = os.path.getsize(item_path)
            print(f"  [FILE] {item} ({size:,} bytes)")
    
    # Check for TensorBoard logs
    tb_dir = os.path.join(run_dir, "tb")
    if os.path.exists(tb_dir):
        tb_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
        print(f"\nFound {len(tb_files)} TensorBoard log files")
        
        # Get file sizes
        for tb_file in tb_files[:3]:  # Show first 3
            size = os.path.getsize(tb_file)
            print(f"  {os.path.basename(tb_file)}: {size:,} bytes")
    
    # Check for checkpoints
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*"))
        print(f"\nFound {len(ckpt_files)} checkpoint files")
        
        # Sort by modification time
        ckpt_files.sort(key=os.path.getmtime)
        for ckpt in ckpt_files[-5:]:  # Show last 5
            mtime = os.path.getmtime(ckpt)
            size = os.path.getsize(ckpt)
            name = os.path.basename(ckpt)
            print(f"  {name}: {size:,} bytes, modified {mtime:.0f}s ago")
    
    # Check for videos if saved
    video_dir = os.path.join(run_dir, "video")
    if os.path.exists(video_dir):
        print(f"\nVideo directory exists: {video_dir}")

def create_placeholder_plots():
    """Create placeholder plots for the presentation."""
    print("\n=== Creating Placeholder Plots ===")
    
    # Create output directory
    os.makedirs("./results", exist_ok=True)
    
    # 1. Learning Curve (placeholder)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # Simulate learning curve
    steps = np.linspace(0, 50000, 100)
    reward = 1 - np.exp(-steps / 10000)  # Simulated learning
    ax1.plot(steps, reward, 'b-', linewidth=2, label='Training Reward')
    ax1.set_xlabel('Environment Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('RL Learning Curve (Placeholder)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.05, 'Actual data will populate after training', 
             transform=ax1.transAxes, fontsize=10, style='italic')
    plt.tight_layout()
    plt.savefig("./results/learning_curve_placeholder.png", dpi=150)
    plt.close()
    
    # 2. Success Rate Comparison (placeholder)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    methods = ['XIRL (Ours)', 'TCN', 'LIFS', 'Goal Classifier', 'Raw']
    success_rates = [0.95, 0.87, 0.82, 0.78, 0.10]  # Paper values
    bars = ax2.bar(methods, success_rates, color=['blue', 'gray', 'gray', 'gray', 'gray'])
    bars[0].set_color('red')  # Highlight XIRL
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Final Success Rate Comparison (Paper Values)')
    ax2.set_ylim([0, 1])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("./results/success_comparison_placeholder.png", dpi=150)
    plt.close()
    
    print("Created placeholder plots in ./results/ directory")
    print("These will be replaced with actual data after training")

if __name__ == "__main__":
    check_training_progress()
    create_placeholder_plots()
    
    print("\n=== Next Steps ===")
    print("1. Let training complete (currently ~14% done)")
    print("2. After completion, we'll:")
    print("   - Extract real metrics from TensorBoard")
    print("   - Generate actual learning curves")
    print("   - Compare with paper results")
    print("   - Create final presentation slides")