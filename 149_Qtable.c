/* 149_Qtable.c */
#include <stdio.h>

#define N_STATES 5
#define N_ACTIONS 2  /* 0=left, 1=right */

int main(void)
{
    /* Initialize Q-table to zeros */
    float Q[N_STATES][N_ACTIONS] = {{0}};
    int s, a;

    printf("Initial Q-table (all zeros):\n\n");
    printf("  state   Q(left)   "
           "Q(right)  best action\n");
    printf("  -----   -------   "
           "--------  -----------\n");
    for (s = 0; s < N_STATES; s++) {
        int best = Q[s][1] > Q[s][0] ? 1 : 0;
        printf("  %3d     %+6.2f    %+6.2f    %s\n",
               s, Q[s][0], Q[s][1], 
               Q[s][0] == Q[s][1] ? "tie" :
               best ? "right" : "left");
    }

    printf("\n  Every Q-value is zero, so the agent\n");
    printf("  knows nothing at all yet.\n");
    printf("  After learning, Q(s, right) should be\n");
    printf("  higher for states left of the goal.\n");

    /* Work the Bellman equation backward from the
       goal to get the values learning should find */
    printf("\n  What the optimal values are, "
           "gamma=0.9\n\n");

    /* Working backward from goal:
       Q(3, right) = -1 + 0.9*10 = 8.0 (reach goal)
       Q(2, right) = -1 + 0.9*8.0 = 6.2
       Q(1, right) = -1 + 0.9*6.2 = 4.58
       Q(0, right) = -1 + 0.9*4.58 = 3.12 */
    float gamma = 0.9f;
    /* Moving right from state 3 lands on the goal
       and pays +10, ending the episode there, so
       that is where the recursion starts. State 4
       is terminal and no action is taken from it. */
    float ideal[N_STATES];
    ideal[N_STATES-2] = 10.0f;
    for (s = N_STATES - 3; s >= 0; s--)
        ideal[s] = -1 + gamma * ideal[s+1];

    printf("  state   ideal Q(right)\n");
    for (s = 0; s < N_STATES - 1; s++)
        printf("  %3d     %+6.2f\n", s, ideal[s]);
    printf("  %3d     terminal, no action\n",
           N_STATES - 1);

    printf("\n  The values fall with distance from "
           "the goal.\n");
    printf("  Learning has to find these from "
           "experience.\n");

    return 0;
}
