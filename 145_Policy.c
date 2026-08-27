/* 145_Policy.c */
#include <stdio.h>
#include <stdlib.h>

#define GRID_SIZE 5
#define GOAL 4
#define N_ACTIONS 2

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

typedef struct {
    int position;
    int done;
}
State;

/* Policy: probability of going right at each
   position */
typedef struct {
    float prob_right[GRID_SIZE];
}
Policy;

static int sample_action(const Policy *p, int pos)
{
    return (randf() < p->prob_right[pos]) ? 1 : 0;
}

static int run_episode(const Policy *p, int verbose)
{
    State s = { .position = 0, .done = 0 };
    int total = 0, step;

    for (step = 0; !s.done && step < 20; step++) {
        int action = sample_action(p, s.position);
        int reward = -1;
        if (action == 1 && s.position < GRID_SIZE - 1)
            s.position++;
        else if (action == 0 && s.position > 0)
            s.position--;
        if (s.position == GOAL) {
            reward = 10;
            s.done = 1;
            }
        total += reward;

        if (verbose)
            printf("    step %d: pos=%d action=%s "
                   "reward=%+d\n",
                   step, s.position, 
                   action ? "right" : "left", reward);
    }
    return total;
}

int main(void)
{
    int i, trial;

    /* Random policy */
    Policy random_p;
    for (i = 0; i < GRID_SIZE; i++)
        random_p.prob_right[i] = 0.5f;

    /* Good policy */
    Policy good_p;
    for (i = 0; i < GRID_SIZE; i++)
        /* almost always go right */
        good_p.prob_right[i] = 0.9f;

    /* Optimal policy */
    Policy optimal_p;
    for (i = 0; i < GRID_SIZE; i++)
        /* always go right */
        optimal_p.prob_right[i] = 1.0f;

    printf("Policies averaged over 1000 "
           "episodes\n\n");

    struct { const char *name; Policy
        *p;
        }
        policies[] = {
        { "Random (50/50)", &random_p },
        { "Good, 90% right", &good_p },
        { "Optimal, 100% right", &optimal_p },
    };

    srand(42);
    for (i = 0; i < 3; i++) {
        float avg = 0;
        for (trial = 0; trial < 1000; trial++)
            avg += run_episode(policies[i].p, 0);
        avg /= 1000;
        printf("  %-22s  avg reward = %+.1f\n",
               policies[i].name, avg);
    }

    printf("\n  The optimal policy reaches the goal "
           "in %d steps.\n", GOAL);
    printf("  A random policy wastes steps on left.\n");
    printf("  The job of RL is finding that policy "
           "from experience.\n");

    return 0;
}
