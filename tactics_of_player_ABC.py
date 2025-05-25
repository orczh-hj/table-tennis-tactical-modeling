"""
Implements three table tennis tactics decision logic of player A, B, C and shot success probability calculation functions.
"""

import random
import numpy as np
import pandas as pd
import torch
from typing import Tuple

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def str_ball(ball: torch.Tensor) -> str:
    """
    Convert ball tensor to human-readable string representation.
    
    Args:
        ball: Input tensor of shape [[spin, line, length, quality]] or [[spin, line, length]]
        
    Returns:
        str: Formatted string describing ball characteristics
    """
    ball = ball.squeeze()
    spin = 'Top spin' if int(ball[0].item()) == 1 else 'Back spin'

    # Determine ball placement line
    if int(ball[1].item()) == -1:
        line = 'forehand'
    elif int(ball[1].item()) == 1:
        line = 'backhand'
    else:
        line = 'middle line'

    # Determine ball length
    if int(ball[2].item()) == 0:
        length = 'short ball'
    elif int(ball[2].item()) == 1:
        length = 'medium length ball'
    else:
        length = 'long ball'

    # Include quality if present
    if len(ball) >= 4:
        quality = ball[3].item()
        return f'{spin} {line} {length}, quality: {quality:.2f}'
    else:
        return f'{spin} {line} {length}'

def compute_probability(
        tcpt: torch.Tensor,
        incoming: torch.Tensor,
        returning: torch.Tensor
) -> float:
    """
    Calculate success probability of a return shot based on player's technical parameters and ball characteristics.
    
    Args:
        tcpt: Player's technical parameters tensor of shape (2, 18, 18)
        incoming: Incoming ball characteristics tensor
        returning: Return ball characteristics tensor
        
    Returns:
        float: Success probability between 0 and 1
    """
    incoming = incoming.squeeze()
    returning = returning.squeeze()
    qr = returning[3]  # Return quality
    qc = incoming[3]   # Incoming quality

    # Handle edge cases
    if qr.item() <= 0.001 or qc.item() >= 0.999:
        return 0.0

    # Calculate index for technical parameter lookup
    returning_idx = (returning[0] + 1) / 2 * 9 + (returning[1] + 1) * 3 + returning[2]
    returning_idx = returning_idx.to(torch.long)

    # Serve handling
    if torch.all(incoming == 0):
        b = tcpt[1, :, returning_idx].mean()
        p = 1 - (1 - b) * qr
    # Return shot handling
    else:
        incoming_idx = (incoming[0] + 1) / 2 * 9 + (incoming[1] + 1) * 3 + incoming[2]
        incoming_idx = incoming_idx.to(torch.long)
        a = tcpt[0, incoming_idx, returning_idx]
        b = tcpt[1, incoming_idx, returning_idx]
        p = 1 - (1 - a) * qc - (1 - b) * qr

    return torch.clip(p, 0, 1).item()

class PlayerA:
    name = 'PlayerA'

    def __init__(self):
        df = pd.read_excel('Athlete TCPTs.xlsx', sheet_name=0, header=0, index_col=0)
        tcpt = torch.as_tensor(np.stack([df.values[:18, :18], df.values[:18, 18:36]]), device=DEVICE, dtype=torch.float)
        self.tcpt = tcpt

    def serve(self) -> Tuple[bool, torch.Tensor]:
        """Execute serve with backspin and random placement."""
        incoming = torch.zeros((1, 4), dtype=torch.float, device=DEVICE)
        balls = [
            [-1, -1, 0, np.clip(random.gauss(0.95, 0.01), 0, 1)],  # 正手短下旋
            [1, 1, 2, np.clip(random.gauss(0.8, 0.1), 0, 1)],  # 反手长上旋
        ]
        ball = torch.as_tensor(random.choice(balls), dtype=torch.float, device=DEVICE)
        ball = ball.unsqueeze(0)
        print(f'{self.name} serve: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball

    def receive(self, incoming: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        """Handle incoming ball with spin-specific responses."""
        incoming = incoming.squeeze()
        spin, line, length, quality = incoming
        ball = torch.zeros((4,), dtype=torch.float, device=DEVICE)

        # Backspin handling
        if spin == -1:
            if length == 2:  # 下旋长球
                # 主要冲正手
                ball[0] = 1
                ball[1] = random.choices([-1, 0, 1], [0.6, 0.1, 0.3])[0]
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.9, 0.05), 0, 1)
            elif length == 1:  # 下旋半出台
                # 搓反手长
                ball[0] = -1
                ball[1] = 1
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.75, 0.1), 0, 1)
            else:
                # 搓长球
                ball[0] = -1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.8, 0.1), 0, 1)
        # Topspin handling
        else:
            if length == 2:  # 上旋长球
                ball[0] = 1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.8, 0.1), 0, 1)
            elif length == 1:  # 上旋半出台
                ball[0] = 1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
            else:
                ball[0] = 1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
        ball = ball.unsqueeze(0)
        print(f'{self.name} return: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball

class PlayerB:
    name = 'PlayerB'

    def __init__(self):
        df = pd.read_excel('Athlete TCPTs.xlsx', sheet_name=1, header=0, index_col=0)
        tcpt = torch.as_tensor(np.stack([df.values[:18, :18], df.values[:18, 18:36]]), device=DEVICE, dtype=torch.float)
        self.tcpt = tcpt

    def serve(self) -> Tuple[bool, torch.Tensor]:
        """Execute topspin serve with varied placement."""
        incoming = torch.zeros((1, 4), dtype=torch.float, device=DEVICE)
        balls = [
            [-1, 0, 0, np.clip(random.gauss(0.9, 0.05), 0, 1)],  # 中路短下旋
            [-1, 1, 2, np.clip(random.gauss(0.9, 0.05), 0, 1)],  # 反手长下旋
        ]
        ball = torch.as_tensor(random.choice(balls), dtype=torch.float, device=DEVICE)
        ball = ball.unsqueeze(0)
        print(f'{self.name} serve: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball

    def receive(self, incoming: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        """Handle incoming ball with consistent deep returns."""
        incoming = incoming.squeeze()
        spin, line, length, quality = incoming
        ball = torch.zeros((4,), dtype=torch.float, device=DEVICE)

        # Backspin handling
        if spin == -1:
            if length == 2:  # 下旋长球
                # 主要冲反手
                ball[0] = 1
                ball[1] = random.choices([-1, 0, 1], [0.3, 0.1, 0.6])[0]
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.9, 0.05), 0, 1)
            elif length == 1:  # 下旋半出台
                # 高调，质量稍低
                ball[0] = 1
                ball[1] = 1
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.85, 0.1), 0, 1)
            else:
                # 搓 或 拧拉、挑打
                ball[0] = random.choices([-1, 1], [0.7, 0.3])[0]
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
        # Topspin handling
        else:
            if length == 2:  # 上旋长球
                ball[0] = 1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.8, 0.1), 0, 1)
            elif length == 1:  # 上旋半出台
                ball[0] = 1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.7, 0.1), 0, 1)
            else:  # 上旋短球
                ball[0] = 1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)

        ball = ball.unsqueeze(0)
        print(f'{self.name} return: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball

class PlayerC:
    name = 'PlayerC'

    def __init__(self):
        df = pd.read_excel('Athlete TCPTs.xlsx', sheet_name=2, header=0, index_col=0)
        tcpt = torch.as_tensor(np.stack([df.values[:18, :18], df.values[:18, 18:36]]), device=DEVICE, dtype=torch.float)
        self.tcpt = tcpt

    def serve(self) -> Tuple[bool, torch.Tensor]:
        """Execute varied spin serves with deep placement."""
        incoming = torch.zeros((1, 4), dtype=torch.float, device=DEVICE)
        balls = [
            [1, 1, 2, np.clip(random.gauss(0.8, 0.1), 0, 1)],  # 反手长上旋
            [-1, 1, 0, np.clip(random.gauss(0.6, 0.1), 0, 1)],  # 反手短下旋
            [1, -1, 0, np.clip(random.gauss(0.8, 0.1), 0, 1)],  # 正手短上旋
        ]
        ball = torch.as_tensor(random.choice(balls), dtype=torch.float, device=DEVICE)

        ball = ball.unsqueeze(0)
        print(f'{self.name} serve: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball

    def receive(self, incoming: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        """Handle incoming ball with safe, high-percentage returns."""
        incoming = incoming.squeeze()
        spin, line, length, quality = incoming
        ball = torch.zeros((4,), dtype=torch.float, device=DEVICE)

        # Backspin handling
        if spin == -1:
            if length == 2:  # 下旋长球
                # 主要拉反手
                ball[0] = 1
                ball[1] = -incoming[1]
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.8, 0.1), 0, 1)
            elif length == 1:  # 下旋半出台
                # 搓反手长
                ball[0] = -1
                ball[1] = 1
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.5, 0.1), 0, 1)
            else:
                # 搓长球
                ball[0] = -1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
        # Topspin handling
        else:
            if length == 2:  # 上旋长球
                ball[0] = 1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                if incoming[1] == 1:
                    ball[3] = np.clip(random.gauss(0.7, 0.1), 0, 1)
                else:
                    ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
            elif length == 1:  # 上旋半出台
                ball[0] = 1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                if incoming[1] == 1:
                    ball[3] = np.clip(random.gauss(0.7, 0.1), 0, 1)
                else:
                    ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
            else:
                ball[0] = 1
                ball[1] = random.choice([-1, 0, 1])
                ball[2] = 1
                ball[3] = np.clip(random.gauss(0.5, 0.1), 0, 1)

        ball = ball.unsqueeze(0)
        print(f'{self.name} return: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball