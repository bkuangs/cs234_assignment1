# Assignment 1 (Problem #4 Only)
**Assignment link:** https://web.stanford.edu/class/cs234/assignments/a1/CS234_A1.pdf  

**Deliverable:** Fill in the `vi_and_pi.py` shell with correct implementations of each concept  


<img width="654" height="209" alt="image" src="https://github.com/user-attachments/assets/9572a748-954e-4568-be40-f548e5ad3762" />


## Key Concepts
**Bellman Backup:** Equation for calculating the disconuted sum of future returns for any given state.  

**Policy Evaluation:** Computing the value function for a given policy given the input MDP. Can think of this like calculating the "scoreboard" for all policies. We start by setting the value of every state to be zero. Then, by following the policies, we calculate the value of each state. As we update the values from zero, it changes the values of other states as well, so we continue to iterate until our change in value is less than the tolerance, meaning we have converged to the true value.  

**Policy Improvment:** Given the value function induced by a given policy, perform policy improvement. Unlike policy evaluation, our "scoreboard" of policy values does not change. Instead, we iterate to find the policy, given that we have calculated the value of each. We iterate over every action within each state and extract the highest value option.  

**Policy Iteration:** Calls `policy_evaluation` and `policy_improvement` to create a map of the best policies with their respective values.

**Value Iteration:** Value iteration differs from policy iteration in that it cares only about the max value at each state and it does not maintain a policy while running. PI evaluates every single policy during each iteration -> less iterations, but longer iterations VI finds the best possible value in each state -> more iterations, but faster iterations. VI effectively combines "Evaluation" and "Improvement" into a single step.  
