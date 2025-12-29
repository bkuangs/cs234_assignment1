### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim
import copy 

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.
    Discounted sum of future returns.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns: Value
    -------
    backup_val: float
    """
    backup_val = R[state, action] + gamma * (
            np.dot(T[state, action], V) # Dot product across entire vector V of possible states
        )
    
    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP.
    We are essentially calculating the "scoreboard" for a given policy.
    We start by setting the value of every state to be zero.
    Then, by following the policies, we calculate the value of each state.
    As we update the values from zero, it changes the values of other states as well, so we continue to iterate.
    Until our change in value is less than the tolerance, meaning we have converged to the true value.

    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns: List of values
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    while True:
        delta = 0  # Change in value
        
        # Update the value of every state
        for s in range(num_states):
            
            val = value_function[s]
            a = policy[s]
            
            new_val = bellman_backup(s, a, R, T, gamma, value_function)
            value_function[s] = new_val
            
            delta = max(delta, abs(val - new_val))
            
        # Stop if the maximum change is smaller than our tolerance
        if delta < tol:
            break

    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement.
    Unlike policy_evaluation, our "scoreboard" of policy values does not change.
    Instead, we iterate to find the policy, given that we have calculated the value of each.
    We iterate over every action within each state and extract the highest value option.

    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns: List of actions
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    for s in range(num_states):

        best_val = -float('inf')  
        best_action = -1

        for a in range(num_actions):
            val = bellman_backup(s, a, R, T, gamma, V_policy)

            if val > best_val:
                best_val = val
                best_action = a
    
    new_policy[s] = best_action

    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """
    Runs policy iteration.
    You should call the policy_evaluation() and policy_improvement() methods to implement this method.

    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns: Map of policies with their values
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    
    while True:
        # Evaluate current policy
        V_policy = policy_evaluation(policy, R, T, gamma, tol=1e-3)

        # Improve the policy based on those values
        new_policy = policy_improvement(policy, R, T, V_policy, gamma)

        # If no improvement, finish loop
        if np.array_equal(new_policy, policy):
            break

        # Otherwise update values and iterate
        policy = new_policy

    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """
    Runs value iteration.
    Value iteration differs from policy iteration in that it cares only about the max value at each state.
    It does not maintain a policy while running.
    PI evaluates every single policy during each iteration -> less iterations, but longer iterations
    VI finds the best possible value in each state -> more iterations, but faster iterations
    VI effectively combines "Evaluation" and "Improvement" into a single step.

    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    
    while True:
        delta = 0
        new_values = np.copy(value_function)

        for s in range(num_states):
            q_values = []

            for a in range(num_actions):

                val = bellman_backup(s, a, R, T, gamma, value_function)
                q_values.append(val)

            best_value = max(q_values)
            best_action = np.argmax(q_values) # Finds the index of the max value

            new_values[s] = best_value
            policy[s] = best_action

            delta = max(delta, abs(value_function[s] - best_value)) # Change in value

        value_function = new_values

        if delta < tol: break

    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'WEAK' # 'WEAK' # 'MEDIUM'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.5
    
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])
