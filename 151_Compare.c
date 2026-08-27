/* 151_Compare.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_STATES 5
#define GOAL 4
#define N_ACTIONS 2
#define MAX_STEPS 20

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}
static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

/* --- Q-learning --- */
static float eval_qlearning(int n_episodes)
{
    float Q[N_STATES][N_ACTIONS] = {{0}};
    float alpha = 0.1f, gamma = 0.9f, epsilon = 0.1f;
    float total = 0;
    int ep, step;

    for (ep = 0; ep < n_episodes; ep++) {
        int s = 0;
        float ep_reward = 0;
        for (step = 0; step < MAX_STEPS; step++) {
            int greedy = Q[s][1] > Q[s][0] ? 1 : 0;
            int a = (randf() < epsilon)
                    ? rand() % 2 : greedy;
            int sn = s;
            if (a==1 && s<N_STATES-1) sn = s+1;
            else if (a==0 && s>0) sn = s-1;
            float r = -1;
            int done = 0;
            if (sn==GOAL) {
                r = 10;
                done = 1;
                }
            float best = Q[sn][0] > Q[sn][1]
                         ? Q[sn][0] : Q[sn][1];
            float target = done ? r : r + gamma * best;
            Q[s][a] += alpha * (target - Q[s][a]);
            ep_reward += r;
            s = sn;
            if (done) break;
        }
        if (ep >= n_episodes - 100) total += ep_reward;
    }
    return total / 100;
}

/* --- REINFORCE --- */
static float eval_reinforce(int n_episodes)
{
    float logits[N_STATES] = {0};
    float total = 0;
    int ep, step;

    for (ep = 0; ep < n_episodes; ep++) {
        int positions[MAX_STEPS], actions[MAX_STEPS];
        float rewards[MAX_STEPS];
        int len = 0, s = 0;
        float ep_reward = 0;

        for (step = 0; step < MAX_STEPS; step++) {
            int a = (randf() < sigmoid(logits[s]))
                ? 1
                : 0;
            positions[len] = s;
            actions[len] = a;
            if (a==1 && s<N_STATES-1) s++;
            else if (a==0 && s>0) s--;
            float r = -1;
            if (s==GOAL) r = 10;
            rewards[len] = r;
            ep_reward += r;
            len++;
            if (s==GOAL) break;
        }

        float G = 0;
        for (step = len-1; step >= 0; step--) {
            G = rewards[step] + 0.99f * G;
            float pr = sigmoid(logits[positions[step]]);
            float grad = actions[step]==1
                ? (1-pr)
                : (-pr);
            logits[positions[step]] += 0.01f * grad * G;
        }

        if (ep >= n_episodes - 100) total += ep_reward;
    }
    return total / 100;
}

int main(void)
{
    int trials[] = { 100, 200, 500, 1000 };
    int n = 4, i;

    printf("Q-Learning against REINFORCE, average "
           "over the last 100\n\n");
    printf("  Episodes   Q-Learning   REINFORCE\n");
    printf("  --------   ----------   ---------\n");

    for (i = 0; i < n; i++) {
        srand(42);
        float ql = eval_qlearning(trials[i]);
        srand(42);
        float rf = eval_reinforce(trials[i]);
        printf("  %5d      %+6.1f       %+6.1f\n",
               trials[i], ql, rf);
    }

    printf("\n  Both converge to the "
           "optimal policy.\n");
    printf("  Q-learning learns values, off-policy.\n");
    printf("  REINFORCE learns a policy, on-policy.\n");
    printf("  For LLMs the policy gradient methods\n");
    printf("  win, because the vocabulary is far\n");
    printf("  too large for a Q-table.\n");

    return 0;
}
