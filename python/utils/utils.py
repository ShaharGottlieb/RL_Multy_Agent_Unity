
import numpy as np
DETECTABLE_OBJECTS = 2
ANGLE_VALUES = DETECTABLE_OBJECTS + 2
N_ANGLES = 11

def preprocess_observations(states, num_agents):
    """Process a agent observation.
    Params
    ======
        states[agent][state]:
        state[0] : direction
        state[1] : velocity magnitude
        states[2:(end-N_ANGLES)] : local observations in fours -
                [ Wall_hit(bool), Player_hit(bool), Other_hit(bool), distance(float) ]
        states[end-N_ANGLES : end] : local observation of next checkpoint in N_ANGLES directions.
        distance is already normalized, and "other_hit" is useless since it captures trigger colliders.
    """
    processed_states = np.zeros([num_agents, DETECTABLE_OBJECTS*N_ANGLES + 1 + N_ANGLES])
                                                            # velocity + player_distances + wall_distances
    for i in range(num_agents):
        processed_states[i][0] = states[i][0] * states[i][1]    # velocity magnitude * direction
        for angle in range(N_ANGLES):

            wall_hit_index = 2 + ANGLE_VALUES*angle
            player_hit_index = wall_hit_index + 1
            distance_index = wall_hit_index + DETECTABLE_OBJECTS + 1
            new_wall_hit_index = DETECTABLE_OBJECTS*angle + 1
            new_player_dist_index = new_wall_hit_index + 1

            processed_states[i][new_wall_hit_index] = states[i][wall_hit_index] * states[i][distance_index]
            processed_states[i][new_player_dist_index] = states[i][player_hit_index] * states[i][distance_index]

        processed_states[i, -N_ANGLES:] = states[i, -N_ANGLES:]

    return processed_states

def processed_state_dim(state_size):
    assert ((state_size-2-N_ANGLES) % ANGLE_VALUES) == 0
    return 1 + DETECTABLE_OBJECTS*N_ANGLES + N_ANGLES
