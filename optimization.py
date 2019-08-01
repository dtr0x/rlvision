# Optimization

def get_last_action(state):
    last_action = state.action_history[:9]
    return last_action.nonzero().item()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: get_last_action(s) != 8,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = [s for s in batch.next_state if get_last_action(s) != 8]
    non_final_img_t, non_final_action_history = state_transform(non_final_next_states)
    
    state_batch = batch.state
    action_batch = torch.tensor(batch.action, device=device)
    reward_batch = torch.cat(batch.reward)
       
    img_t, action_history = state_transform(state_batch)
    
    actions = policy_net(img_t, action_history)

    state_action_values = policy_net(img_t, action_history).gather(1, action_batch.view(-1, 1))

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_img_t, non_final_action_history).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp(-1, 1)
    optimizer.step()