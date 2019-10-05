
import numpy as np
DETECTABLE_OBJECTS = 2
ANGLE_VALUES = DETECTABLE_OBJECTS + 2

def preprocess_observations(states, num_agents):
    """Process a agent observation.
    Params
    ======
        states[agent][state]:
        state[0] : direction
        state[1] : velocity magnitude
        states[2:end] : local observations in fours -
                [ Wall_hit(bool), Player_hit(bool), Other_hit(bool), distance(float) ]
        distance is already normalized, and "other_hit" is useless since it captures trigger colliders.
    """
    num_of_angles = (len(states[0]) - 2) // ANGLE_VALUES
    processed_states = np.zeros([num_agents, DETECTABLE_OBJECTS*num_of_angles + 1])
                                                            # velocity + player_distances + wall_distances
    for i in range(num_agents):
        processed_states[i][0] = states[i][0] * states[i][1]    # velocity magnitude * direction
        for angle in range(num_of_angles):

            wall_hit_index = 2 + ANGLE_VALUES*angle
            player_hit_index = wall_hit_index + 1
            distance_index = wall_hit_index + DETECTABLE_OBJECTS + 1
            new_wall_hit_index = DETECTABLE_OBJECTS*angle + 1
            new_player_dist_index = new_wall_hit_index + 1

            processed_states[i][new_wall_hit_index] = states[i][wall_hit_index] * states[i][distance_index]
            processed_states[i][new_player_dist_index] = states[i][player_hit_index] * states[i][distance_index]

    return processed_states

def processed_state_dim(state_size):
    num_of_angles = (state_size - 2) // ANGLE_VALUES
    assert ((state_size-2) % ANGLE_VALUES) == 0
    return 1 + DETECTABLE_OBJECTS*num_of_angles
