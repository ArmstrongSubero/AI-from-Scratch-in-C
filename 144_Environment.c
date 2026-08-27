/* 144_Environment.c */
#include <stdio.h>

#define GRID_SIZE 5
#define GOAL 4

typedef struct {
    int position;
    int done;
}
State;

typedef struct {
    int reward;
    State next_state;
}
StepResult;

/* Actions: 0 = left, 1 = right */
static StepResult env_step(State s, int action)
{
    StepResult r;
    r.next_state = s;
    r.reward = -1;  /* step penalty */

    if (action == 1 && s.position < GRID_SIZE - 1)
        r.next_state.position++;
    else if (action == 0 && s.position > 0)
        r.next_state.position--;

    if (r.next_state.position == GOAL) {
        r.reward = 10;
        r.next_state.done = 1;
    }

    return r;
}

int main(void)
{
    State s = { .position = 0, .done = 0 };
    int total_reward = 0;
    int step;

    printf("Grid world, 5 positions, goal at %d\n",
           GOAL);
    printf("  Actions: 0=left, 1=right\n");
    printf("  Rewards: -1 per step, +10 at goal\n\n");

    /* Hardcoded policy: always go right */
    printf("  step  pos  action  reward  total\n");
    printf("  ----  ---  ------  ------  -----\n");
    for (step = 0; !s.done && step < 10; step++) {
        int action = 1;  /* always right */
        StepResult r = env_step(s, action);
        total_reward += r.reward;
        printf("  %3d    %d    right   %+3d    %+3d\n",
               step, s.position, r.reward, 
                   total_reward);
        s = r.next_state;
    }

    /* The move that lands on the goal pays +10
       in place of the -1, so a %d step run costs
       %d penalties and collects 10 */
    printf("\n  Total reward: %d\n", total_reward);
    printf("  Best possible: %d steps, %d at -1, "
           "then +10 = %d\n",
           GOAL, GOAL - 1, -(GOAL - 1) + 10);

    return 0;
}
