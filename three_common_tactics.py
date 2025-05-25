"""
Implements three table tennis tactics decision logic and shot success probability calculation functions.
"""

import random
import numpy as np
import torch

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

class Tactic1:
    """Backspin-oriented attack tactic focusing on spin variation and placement."""
    name = 'Backspin attack'

    def __init__(self, tcpt: torch.Tensor):
        self.tcpt = tcpt

    def serve(self) -> tuple[bool, torch.Tensor]:
        """Execute serve with backspin and random placement."""
        incoming = torch.zeros((1, 4), dtype=torch.float, device=DEVICE)
        ball = torch.zeros((4,), dtype=torch.float, device=DEVICE)

        # Backspin serve characteristics
        ball[0] = -1  # Spin type: backspin
        ball[1] = random.choice([-1, 0, 1])  # Random line
        ball[2] = random.choices([-1, 0, 1], [0.6, 0.2, 0.2])[0]  # Mostly short
        ball[3] = np.clip(random.gauss(0.8, 0.1), 0, 1)  # High quality

        ball = ball.unsqueeze(0)
        print(f'{self.name} serve: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball

    def receive(self, incoming: torch.Tensor) -> tuple[bool, torch.Tensor]:
        """Handle incoming ball with spin-specific responses."""
        incoming = incoming.squeeze()
        spin, line, length, quality = incoming
        ball = torch.zeros((4,), dtype=torch.float, device=DEVICE)

        # Backspin handling
        if spin == -1:
            if length == 2:  # Long backspin
                ball[0] = 1  # Topspin attack
                ball[1] = -line  # Cross-court
                ball[2] = 1  # Long return
                ball[3] = np.clip(random.gauss(0.9, 0.1), 0, 1)
            elif length == 1:  # Medium backspin
                ball[0] = 1  # Topspin
                ball[1] = -line  # Cross-court
                ball[2] = 1  # Long return
                ball[3] = np.clip(random.gauss(0.5, 0.1), 0, 1)
            else:  # Short backspin
                ball[0] = -1  # Push return
                ball[1] = random.choice([-1, 0, 1])  # Random placement
                ball[2] = 1  # Long return
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
        # Topspin handling
        else:
            if length == 2:  # Long topspin
                ball[0] = 1  # Topspin counter
                ball[1] = -line  # Cross-court
                ball[2] = 1  # Long return
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
            else:  # Medium/short topspin
                ball[0] = 1  # Topspin
                ball[1] = random.choice([-1, 0, 1])  # Varied placement
                ball[2] = 1  # Long return
                ball[3] = np.clip(random.gauss(0.4, 0.1), 0, 1)

        ball = ball.unsqueeze(0)
        print(f'{self.name} return: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball

class Tactic2:
    """Rally-oriented tactic emphasizing consistent shot placement and spin control."""
    name = 'Rally'

    def __init__(self, tcpt: torch.Tensor):
        self.tcpt = tcpt

    def serve(self) -> tuple[bool, torch.Tensor]:
        """Execute topspin serve with varied placement."""
        incoming = torch.zeros((1, 4), dtype=torch.float, device=DEVICE)
        ball = torch.zeros((4,), dtype=torch.float, device=DEVICE)

        # Topspin serve characteristics
        ball[0] = 1  # Spin type: topspin
        ball[1] = random.choice([-1, 0, 1])  # Random line
        ball[2] = random.choices([-1, 0, 1], [0.4, 0.3, 0.3])[0]  # Balanced length
        ball[3] = np.clip(random.gauss(0.8, 0.1), 0, 1)  # High quality

        ball = ball.unsqueeze(0)
        print(f'{self.name} serve: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball

    def receive(self, opo_tcpt: torch.Tensor, incoming: torch.Tensor) -> tuple[bool, torch.Tensor]:
        """Handle incoming ball with consistent deep returns."""
        incoming = incoming.squeeze()
        spin, line, length, quality = incoming
        ball = torch.zeros((4,), dtype=torch.float, device=DEVICE)

        # Backspin handling
        if spin == -1:
            if length == 2:  # Long backspin
                ball[0] = 1  # Topspin drive
                ball[1] = -line  # Cross-court
                ball[2] = 1  # Deep return
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
            else:  # Medium/short backspin
                ball[0] = -1  # Push return
                ball[1] = random.choice([-1, 0, 1])  # Random placement
                ball[2] = 1  # Deep return
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
        # Topspin handling
        else:
            if length == 2:  # Long topspin
                ball[0] = 1  # Counter-drive
                ball[1] = random.choices([-1, 0, 1], [0.2, 0.2, 0.6])[0]  # Favor backhand
                ball[2] = 1  # Deep return
                ball[3] = np.clip(random.gauss(0.8, 0.1), 0, 1)
            else:  # Medium/short topspin
                ball[0] = 1  # Topspin
                ball[1] = -line  # Cross-court
                ball[2] = 1  # Deep return
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)

        ball = ball.unsqueeze(0)
        print(f'{self.name} return: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball

class Tactic3:
    """Defensive tactic focusing on high-percentage returns and opponent error forcing."""
    name = 'Defense'

    def __init__(self, tcpt: torch.Tensor):
        self.tcpt = tcpt

    def serve(self) -> tuple[bool, torch.Tensor]:
        """Execute varied spin serves with deep placement."""
        incoming = torch.zeros((1, 4), dtype=torch.float, device=DEVICE)
        ball = torch.zeros((4,), dtype=torch.float, device=DEVICE)

        # Mixed spin serve characteristics
        ball[0] = random.choice([-1, 1])  # Random spin
        ball[1] = random.choice([-1, 0, 1])  # Random line
        ball[2] = random.choices([0, 1], [0.2, 0.8])[0]  # Mostly deep
        ball[3] = np.clip(random.gauss(0.8, 0.1), 0, 1)  # High quality

        ball = ball.unsqueeze(0)
        print(f'{self.name} serve: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball

    def receive(self, incoming: torch.Tensor) -> tuple[bool, torch.Tensor]:
        """Handle incoming ball with safe, high-percentage returns."""
        incoming = incoming.squeeze()
        spin, line, length, quality = incoming
        ball = torch.zeros((4,), dtype=torch.float, device=DEVICE)

        # Backspin handling
        if spin == -1:
            if length == 2:  # Long backspin
                ball[0] = 1  # Safe topspin
                ball[1] = 1  # Backhand target
                ball[2] = 1  # Deep return
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
            else:  # Medium/short backspin
                ball[0] = -1  # Push return
                ball[1] = random.choice([-1, 0, 1])  # Random placement
                ball[2] = 1  # Deep return
                ball[3] = np.clip(random.gauss(0.7, 0.1), 0, 1)
        # Topspin handling
        else:
            if length in [2, 1]:  # Long/medium topspin
                ball[0] = 1  # Counter-drive
                ball[1] = random.choices([-1, 0, 1], [0.2, 0.2, 0.6])[0]  # Favor backhand
                ball[2] = 1  # Deep return
                ball[3] = np.clip(random.gauss(0.6, 0.1), 0, 1)
            else:  # Short topspin
                ball[0] = 1  # Topspin flip
                ball[1] = -line  # Cross-court
                ball[2] = 1  # Deep return
                ball[3] = np.clip(random.gauss(0.5, 0.1), 0, 1)

        ball = ball.unsqueeze(0)
        print(f'{self.name} return: {str_ball(ball)}')

        success_prob = compute_probability(self.tcpt, incoming, ball)
        done = random.random() > success_prob
        if done:
            print(f'{self.name} miss')
        return done, ball