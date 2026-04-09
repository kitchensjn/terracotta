import numpy as np

import scipy as sp


def obtain_steady_state_with_matrix_exponential(Q, max_t=100):
    """
    Solve the defining differential equation until it converges.
    
    - Q: the transition matrix
    - max_t: the maximum time for which the differential equation is solved at each attempt.
    """
    
    dimension = Q.shape[0]
    state = np.ones(dimension) / dimension
    
    while not is_steady_state(state=state, Q=Q):
        state = state @ sp.linalg.expm(Q * max_t)
    
    return state

def get_transition_matrix(
    number_of_servers,
    total_capacity,
    arrival_rate,
    service_rate
):
    """
    Obtain Q for an M/M/c/K queue with given parameters:
    
    - number_of_servers: c
    - total_capacity: K
    - arrival_rate: lambda
    - service_rate: mu
    """
    Q = np.zeros((total_capacity + 1, total_capacity + 1))
    
    for i in range(total_capacity + 1):
        total_rate = 0
        
        if i < total_capacity:
            Q[i, i + 1] = arrival_rate
            total_rate += Q[i, i + 1]
        
        if i > 0:
            Q[i, i - 1] = min(i, number_of_servers) * service_rate
            total_rate += Q[i, i - 1]
        
        Q[i, i] = - total_rate
        
    return Q


# A test
expected_Q_MM23 = np.array([[-1.,  1.,  0.,  0.],
                            [ 2., -3.,  1.,  0.],
                            [ 0.,  4., -5.,  1.],
                            [ 0.,  0.,  4., -4.]])
assert np.array_equal(
    get_transition_matrix(
        number_of_servers=2, 
        total_capacity=3, 
        arrival_rate=1, 
        service_rate=2),
    expected_Q_MM23
)


Q = get_transition_matrix(number_of_servers=1, total_capacity=3, arrival_rate=1, service_rate=2)

def is_steady_state(state, Q):
    """
    Returns a boolean as to whether a given state is a steady 
    state of the Markov chain corresponding to the matrix Q
    """
    return np.allclose((state @ Q), 0)

state = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
assert not is_steady_state(state=state, Q=Q)

state = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
print(is_steady_state(state=state, Q=Q))

state = obtain_steady_state_with_matrix_exponential(Q=Q, max_t=1)
print(state)