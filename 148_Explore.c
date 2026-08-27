/* 148_Explore.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

/* Epsilon-greedy. With probability epsilon take a
   random action.
   Otherwise, take the best known action. */

#define N_ACTIONS 3

int main(void)
{
    /* Three slot machines (bandits) with unknown
       payouts */
    float true_rewards[N_ACTIONS] = { 1.0f, 
        2.0f, 0.5f };
    float estimated[N_ACTIONS] = { 0, 0, 0 };
    int counts[N_ACTIONS] = { 0, 0, 0 };
    float total = 0;
    int t;

    float epsilon = 0.1f;

    srand(42);

    printf("Epsilon-greedy exploration, "
           "epsilon=%.1f\n\n", epsilon);

    for (t = 0; t < 1000; t++) {
        int action;
        if (randf() < epsilon) {
            /* Explore: random action */
            action = rand() % N_ACTIONS;
        }
        else {
            /* Exploit: best estimated action */
            action = 0;
            int i;
            for (i = 1; i < N_ACTIONS; i++)
                if (estimated[i] > estimated[action])
                    action = i;
        }

        /* Get reward (with noise) */
        float reward = true_rewards[action]
                       + (randf() - 0.5f);
        total += reward;

        /* Update estimate */
        counts[action]++;
        estimated[action] += 
            (reward - estimated[action])
                / counts[action];

        if ((t+1) % 200 == 0) {
            printf("  t=%4d  total=%.0f  "
                   "estimates=[%.2f, %.2f, %.2f]  "
                   "counts=[%d, %d, %d]\n",
                   t+1, total, 
                   estimated[0], estimated[1], 
                       estimated[2], 
                   counts[0], counts[1], counts[2]);
        }
    }

    printf("\n  Action 1 has the highest true "
           "reward at %.1f.\n", true_rewards[1]);
    printf("  The agent found that and exploits it.\n");
    printf("  Epsilon=%.1f means %.0f%% of actions "
           "are exploration.\n",
           epsilon, epsilon * 100);

    return 0;
}
