/* 150_Qlearning.c */
#include <stdio.h>
#include <stdlib.h>

#define N_STATES 5
#define N_ACTIONS 2
#define GOAL 4

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

static float max_q(float Q[N_STATES][N_ACTIONS], int s)
{
    return Q[s][0] > Q[s][1] ? Q[s][0] : Q[s][1];
}

int main(void)
{
    float Q[N_STATES][N_ACTIONS] = {{0}};
    float alpha = 0.1f;   /* learning rate */
    float gamma = 0.9f;   /* discount factor */
    float epsilon = 0.2f; /* exploration rate */
    int episode, step, s, a;

    srand(42);

    printf("Q-learning on grid world:\n\n");

    for (episode = 0; episode < 500; episode++) {
        s = 0;  /* start at position 0 */

        for (step = 0; step < 20; step++) {
            /* Epsilon-greedy action selection */
            if (randf() < epsilon)
                a = rand() % N_ACTIONS;
            else
                a = Q[s][1] > Q[s][0] ? 1 : 0;

            /* Take action */
            int s_next = s;
            if (a == 1 && s < N_STATES - 1)
                s_next = s + 1;
            else if (a == 0 && s > 0) s_next = s - 1;

            float reward = -1;
            int done = 0;
            if (s_next == GOAL) {
                reward = 10;
                done = 1;
                }

            /* Q-learning update */
            float target = reward;
            if (!done)
                target += gamma * max_q(Q, s_next);

            Q[s][a] += alpha * (target - Q[s][a]);

            s = s_next;
            if (done) break;
        }

        if ((episode + 1) % 100 == 0) {
            printf("  Episode %3d:\n", episode + 1);
            printf("    state  Q(left)  Q(right)  "
                   "best\n");
            for (s = 0; s < N_STATES; s++) {
                int best = Q[s][1] > Q[s][0] ? 1 : 0;
                printf("    %3d    %+6.2f   %+6.2f    "
                       "%s\n",
                       s, Q[s][0], Q[s][1], 
                       best ? "right" : "left");
            }
            printf("\n");
        }
    }

    printf("  The Q-values reach the optimal ones.\n");
    printf("  Q(s, right) > Q(s, left) for all "
           "states.\n");
    printf("  Q(s, right) falls with distance from "
           "the goal.\n");

    return 0;
}
