#!/usr/bin/env python3
"""
Quick test to verify batch update format
"""
import torch as th

# Simulate batch update format
batch_size = 1
num_followup_agents = 1

# Test data
initial_agent = 2
followup_target = 3

# Format according to Wolfpack pattern
init_agent_store = [initial_agent]  # [agent_id]
followup_store = [followup_target]  # [agent_id] for each num_followup_agents slot

# Pad if needed
while len(followup_store) < num_followup_agents:
    followup_store.append(5)  # n_agents placeholder

# Convert to tensors (what batch.update does internally)
init_tensor = th.tensor([init_agent_store], dtype=th.long)  # [1, 1]
followup_tensor = th.tensor([followup_store], dtype=th.long)  # [1, 1]

print(f"initial_agent shape: {init_tensor.shape} (expected: [1, 1])")
print(f"followup_agents shape: {followup_tensor.shape} (expected: [1, 1])")
print(f"initial_agent value: {init_tensor}")
print(f"followup_agents value: {followup_tensor}")

assert init_tensor.shape == (1, 1), f"Wrong initial_agent shape: {init_tensor.shape}"
assert followup_tensor.shape == (1, num_followup_agents), f"Wrong followup_agents shape: {followup_tensor.shape}"
print("\n✓ All format checks passed!")
