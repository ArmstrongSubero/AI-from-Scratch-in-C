/* 147_Reinforce.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID_SIZE 5
#define GOAL 4
#define N_ACTIONS 2

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

/* Policy: logit for "go right" at each position */
float policy_logits[GRID_SIZE];

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

static int sample_action(int pos)
{
    float prob_right = sigmoid(policy_logits[pos]);
    return (randf() < prob_right) ? 1 : 0;
}

/* Store an episode */
#define MAX_STEPS 30
static int ep_positions[MAX_STEPS];
static int ep_actions[MAX_STEPS];
static float ep_rewards[MAX_STEPS];
static int ep_len;

static float run_episode(void)
{
    int pos = 0;
    float total = 0;
    ep_len = 0;

    int step;
    for (step = 0; step < MAX_STEPS; step++) {
        int action = sample_action(pos);
        ep_positions[ep_len] = pos;
        ep_actions[ep_len] = action;

        float reward = -1;
        if (action == 1 && pos < GRID_SIZE - 1) pos++;
        else if (action == 0 && pos > 0) pos--;
        if (pos == GOAL) {
            reward = 10;
            }

        ep_rewards[ep_len] = reward;
        total += reward;
        ep_len++;

        if (pos == GOAL) break;
    }
    return total;
}

static void update_policy(float lr)
{
    /* Compute returns (backward) */
    float returns[MAX_STEPS];
    float G = 0;
    int t;

    for (t = ep_len - 1; t >= 0; t--) {
        G = ep_rewards[t] + 0.99f * G;
        returns[t] = G;
    }

    /* Policy gradient: increase prob of good actions */
    for (t = 0; t < ep_len; t++) {
        int pos = ep_positions[t];
        int action = ep_actions[t];
        float prob_right = sigmoid(policy_logits[pos]);

        /* Gradient of log(prob) w.r.t. logit */
        float grad;
        if (action == 1)
            /* d log(sigma(x)) / dx */
            grad = 1.0f - prob_right;
        else
            /* d log(1-sigma(x)) / dx */
            grad = -prob_right;

        /* Scale by return (REINFORCE) */
        policy_logits[pos] += lr * grad * returns[t];
    }
}

int main(void)
{
    int i, ep;
    srand(42);

    /* Initialize policy logits to 0 (50/50 random) */
    for (i = 0; i < GRID_SIZE; i++)
        policy_logits[i] = 0.0f;

    printf("REINFORCE policy gradient training:\n\n");
    printf("  episode   avg_reward   policy, prob "
           "right per pos\n");
    printf("  -------   ----------   -------------"
           "--------------\n");

    for (ep = 0; ep < 500; ep++) {
        float total = 0;
        int batch = 10;
        int b;

        for (b = 0; b < batch; b++) {
            total += run_episode();
            update_policy(0.01f);
        }

        if ((ep + 1) % 100 == 0) {
            printf("  %5d     %+6.1f       [",
                   ep + 1, total / batch);
            for (i = 0; i < GRID_SIZE; i++)
                printf("%.2f%s",
                    sigmoid(policy_logits[i]), 
                       i < GRID_SIZE - 1 ? ", " : "");
            printf("]\n");
        }
    }

    printf("\n  The policy converges to a high "
           "prob_right at every\n");
    printf("  position, which is the "
           "optimal policy.\n");
    printf("  No one told the agent to go right. "
           "It found\n");
    printf("  that out from reward signals alone.\n");

    return 0;
}
