import numpy as np

from c_elegans_independent_model_training import calc_elongation_factor


def p_synapse_is_formed(spls_across_dev, smi, beta, developmental_ages, time_passed_from_given_state,
                        time_of_given_state, norm_length, given_past_state=0, time_step=10):
    # Initialize the matrix of pmf of the number of synapses across time (the i,j-th entry of
    # `num_synapses_pmf_across_time` is the probability to see j synapses at timestep i).
    num_time_steps_for_calculation = int(time_passed_from_given_state / time_step)
    max_num_synapses_at_the_end = given_past_state + num_time_steps_for_calculation
    num_synapses_pmf_across_time = np.zeros((num_time_steps_for_calculation + 1, max_num_synapses_at_the_end + 1))
    num_synapses_pmf_across_time[0, given_past_state] = 1

    cur_developmental_stage = 0
    spl = spls_across_dev[cur_developmental_stage]
    for step in range(1, num_time_steps_for_calculation + 1):
        # The age of the worm n time steps after the given state of the synapse.
        cur_worm_age = time_of_given_state + step * time_step
        while cur_worm_age > developmental_ages[cur_developmental_stage]:
            cur_developmental_stage += 1
            spl = spls_across_dev[cur_developmental_stage]

        elongation_factor = calc_elongation_factor(cur_worm_age)
        formation_prob = spl * np.exp(-beta * norm_length * elongation_factor)
        for syn_count in range(max_num_synapses_at_the_end + 1):
            # The number of synapses can be changed only by 1 at each step, so the probability to be further from a
            # given state than the number of steps passed from its corresponding time is 0.
            if abs(syn_count - given_past_state) > step:
                continue

            # The probability to have no synapses is the probability of having no synapses at the previous stage, and
            # not forming any now.
            elif syn_count == 0:
                num_synapses_pmf_across_time[step, syn_count] = num_synapses_pmf_across_time[step - 1, syn_count] * (
                        1 - smi)

            # Handel the edge case of the maximal number of synapses - there is no entry holding the probability for
            # more synapses in the matrix. The probability is to have 1 less at the previous step, to form now, and to
            # not erase anything.
            elif syn_count == max_num_synapses_at_the_end:
                num_synapses_pmf_across_time[step, syn_count] = (
                        num_synapses_pmf_across_time[step - 1, syn_count - 1] * formation_prob * (1 - smi))

            else:
                # Don't prune and don't form
                prob_to_stay_the_same = num_synapses_pmf_across_time[step - 1, syn_count] * (1 - smi) * (
                        1 - formation_prob)

                # Form a connection
                prob_to_add_connection = num_synapses_pmf_across_time[step - 1, syn_count - 1] * formation_prob
                # If there were connections before, don't prune them
                if syn_count - 1 > 0:
                    prob_to_add_connection *= (1 - smi)

                prob_to_remove_connection = num_synapses_pmf_across_time[step - 1, syn_count + 1] * smi * (
                        1 - formation_prob)

                num_synapses_pmf_across_time[
                    step, syn_count] = prob_to_stay_the_same + prob_to_add_connection + prob_to_remove_connection

    # The probability to have at least one synapse at the final step
    return num_synapses_pmf_across_time[-1, 1:].sum()
