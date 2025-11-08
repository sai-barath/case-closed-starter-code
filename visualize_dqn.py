"""
Visualizations for DQN Training System

This script creates helpful visualizations:
1. Network architecture diagram
2. Training flow diagram
3. State representation visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def plot_network_architecture():
    """Visualize the DQN network architecture."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'DQN Network Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Input layer
    input_box = FancyBboxPatch((0.5, 7), 1.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 7.75, 'Input State\n7×18×20\n+8 features', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Conv layers
    conv_boxes = [
        (2.5, 7.5, 'Conv2D\n32 filters\n3×3'),
        (4, 7.5, 'Conv2D\n64 filters\n3×3'),
        (5.5, 7.5, 'Conv2D\n64 filters\n3×3'),
    ]
    
    for x, y, text in conv_boxes:
        box = FancyBboxPatch((x, y), 1, 1, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='green', facecolor='lightgreen', linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.5, y + 0.5, text, 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Flatten
    flatten_box = FancyBboxPatch((7, 7.5), 1, 1, 
                                 boxstyle="round,pad=0.05", 
                                 edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(flatten_box)
    ax.text(7.5, 8, 'Flatten +\nConcat', 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    # FC layers
    fc_boxes = [
        (3, 5.5, 'FC Layer\n512 units'),
        (5, 5.5, 'FC Layer\n256 units'),
        (7, 5.5, 'Output\n4 Q-values'),
    ]
    
    for x, y, text in fc_boxes:
        color = 'lightcoral' if 'Output' in text else 'lavender'
        edge = 'red' if 'Output' in text else 'purple'
        box = FancyBboxPatch((x, y), 1.5, 1, 
                             boxstyle="round,pad=0.05", 
                             edgecolor=edge, facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.75, y + 0.5, text, 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrows = [
        ((2, 7.75), (2.5, 8)),
        ((3.5, 8), (4, 8)),
        ((5, 8), (5.5, 8)),
        ((6.5, 8), (7, 8)),
        ((7.5, 7.5), (5.75, 6.5)),
        ((4.5, 6), (5, 6)),
        ((6.5, 6), (7, 6)),
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='gray')
        ax.add_patch(arrow)
    
    # Annotations
    annotations = [
        (1.25, 6, 'Spatial features:\n• My trail\n• Opp trail\n• Walls\n• Positions\n• Valid moves\n• Danger zones', 'left'),
        (7.5, 9.5, 'Extra features:\n• Boosts\n• Trail lengths\n• Distance\n• Direction', 'left'),
        (7.75, 4.5, 'Actions:\nUP, DOWN,\nLEFT, RIGHT', 'center'),
    ]
    
    for x, y, text, align in annotations:
        ax.text(x, y, text, ha=align, va='top', fontsize=7, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('dqn_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: dqn_architecture.png")
    plt.close()


def plot_training_flow():
    """Visualize the training process flow."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'DQN Training Flow', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Process boxes
    processes = [
        (5, 8.5, 'Initialize\nPolicy & Target\nNetworks', 'lightblue'),
        (5, 7.5, 'Reset\nEnvironment', 'lightgreen'),
        (2, 6.5, 'Agent 1\nSelect Action\n(ε-greedy)', 'lightyellow'),
        (8, 6.5, 'Agent 2\nSelect Action\n(ε-greedy)', 'lightyellow'),
        (5, 5.5, 'Execute Actions\nGet Rewards', 'lightcoral'),
        (5, 4.5, 'Store Experience\nin Replay Buffer', 'lavender'),
        (5, 3.5, 'Sample Batch\nCompute Loss', 'lightgreen'),
        (5, 2.5, 'Update Policy\nNetwork', 'lightyellow'),
        (5, 1.5, 'Update Target\n(every N steps)', 'lightblue'),
    ]
    
    for x, y, text, color in processes:
        box = FancyBboxPatch((x - 0.75, y - 0.3), 1.5, 0.6, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows
    arrows = [
        ((5, 8.2), (5, 7.8)),
        ((5, 7.2), (2, 6.8)),
        ((5, 7.2), (8, 6.8)),
        ((2, 6.2), (5, 5.8)),
        ((8, 6.2), (5, 5.8)),
        ((5, 5.2), (5, 4.8)),
        ((5, 4.2), (5, 3.8)),
        ((5, 3.2), (5, 2.8)),
        ((5, 2.2), (5, 1.8)),
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=2, color='gray')
        ax.add_patch(arrow)
    
    # Loop back arrow
    loop_arrow = FancyArrowPatch((5.75, 1.5), (6.5, 6.5),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2, color='red', linestyle='dashed',
                                connectionstyle="arc3,rad=.5")
    ax.add_patch(loop_arrow)
    ax.text(7.5, 4, 'Game Loop', color='red', fontsize=9, fontweight='bold')
    
    # Decision diamond
    diamond_points = [(5, 1), (5.5, 0.5), (5, 0), (4.5, 0.5)]
    diamond = mpatches.Polygon(diamond_points, closed=True, 
                              edgecolor='black', facecolor='orange', linewidth=2)
    ax.add_patch(diamond)
    ax.text(5, 0.5, 'Done?', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Final arrows
    FancyArrowPatch((5, 1.2), (5, 1), arrowstyle='->', mutation_scale=15, 
                   linewidth=2, color='gray')
    ax.text(3.5, 0.5, 'No', fontsize=8, color='red')
    ax.text(5.5, 0.2, 'Yes → Next Episode', fontsize=8, color='green')
    
    plt.tight_layout()
    plt.savefig('dqn_training_flow.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: dqn_training_flow.png")
    plt.close()


def plot_state_representation():
    """Visualize the state representation channels."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('DQN State Representation (7 Spatial Channels + 8 Features)', 
                 fontsize=16, fontweight='bold')
    
    # Create sample board
    height, width = 18, 20
    
    # Channel titles and descriptions
    channels = [
        ('My Trail', 'Positions I\'ve visited'),
        ('Opponent Trail', 'Positions opponent visited'),
        ('Walls/Occupied', 'Blocked cells'),
        ('My Position', 'Current position'),
        ('Opponent Position', 'Opponent current pos'),
        ('Valid Moves', 'Safe cells to move to'),
        ('Danger Zones', 'Opponent might move here'),
        ('Extra Features', 'Boosts, lengths, distance,\ndirection (8 values)'),
    ]
    
    for idx, (ax, (title, desc)) in enumerate(zip(axes.flat, channels)):
        if idx < 7:
            # Create sample spatial data
            data = np.zeros((height, width))
            
            if idx == 0:  # My trail
                data[5:10, 8:12] = 1
            elif idx == 1:  # Opponent trail
                data[12:15, 14:17] = 1
            elif idx == 2:  # Walls
                data[5:10, 8:12] = 0.5
                data[12:15, 14:17] = 0.5
            elif idx == 3:  # My position
                data[9, 11] = 1
            elif idx == 4:  # Opponent position
                data[14, 16] = 1
            elif idx == 5:  # Valid moves
                data[8:11, 10:13] = 0.7
            elif idx == 6:  # Danger zones
                data[13:16, 15:18] = 0.5
            
            im = ax.imshow(data, cmap='Blues' if idx < 5 else 'Reds', 
                          aspect='auto', interpolation='nearest')
            ax.set_title(f'{title}\n({desc})', fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # Extra features visualization
            ax.axis('off')
            feature_names = [
                'My Boosts: 0.67',
                'Opp Boosts: 0.33',
                'Turn Count: 0.15',
                'My Length: 0.18',
                'Opp Length: 0.12',
                'Distance: 0.35',
                'Direction X: 1.0',
                'Direction Y: 0.0',
            ]
            ax.text(0.5, 0.5, title + '\n\n' + '\n'.join(feature_names),
                   ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('dqn_state_representation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: dqn_state_representation.png")
    plt.close()


def plot_reward_structure():
    """Visualize the reward structure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    events = ['Survival\n(per step)', 'Win', 'Loss', 'Draw']
    rewards = [1.0, 100.0, -100.0, -50.0]
    colors = ['lightgreen', 'green', 'red', 'orange']
    
    bars = ax.bar(events, rewards, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{reward:+.1f}',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=12, fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('Reward Value', fontsize=12, fontweight='bold')
    ax.set_title('DQN Reward Structure', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_reward_structure.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: dqn_reward_structure.png")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("Generating DQN System Visualizations")
    print("="*70)
    print()
    
    try:
        plot_network_architecture()
        plot_training_flow()
        plot_state_representation()
        plot_reward_structure()
        
        print()
        print("="*70)
        print("✅ All visualizations generated successfully!")
        print("="*70)
        print("\nGenerated files:")
        print("  • dqn_architecture.png - Network structure")
        print("  • dqn_training_flow.png - Training process")
        print("  • dqn_state_representation.png - Input encoding")
        print("  • dqn_reward_structure.png - Reward system")
        print()
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        print("Note: Requires matplotlib. Install with: pip install matplotlib")
