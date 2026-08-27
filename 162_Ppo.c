/* 162_Ppo.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


/* Simplified PPO for a discrete action space.
   The key idea: clipped surrogate objective.

   ratio = pi_new(a|s) / pi_old(a|s)
   advantage = reward - baseline
   L = min(ratio * adv, 
           clip(ratio, 1-eps, 1+eps) * adv)

   The clipping prevents the policy from changing
   too much
   in a single update. */

static float clip(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

#define N_ACTIONS 4
#define N_INNER 4
#define N_STEPS 10
#define N_SAMPLES 8

static void softmax(float *x, int n)
{
    float mx = -1e9f, s = 0;
    int i;
    for (i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    for (i = 0; i < n; i++) x[i] /= s;
}

int main(void)
{
    /* Policy logits (what we are optimizing) */
    float logits[N_ACTIONS] = { 0.5f, 0.3f, 
        -0.2f, 0.1f };
    /* Old policy (frozen for ratio computation) */
    float old_logits[N_ACTIONS];

    /* Rewards for each action (from the reward
       model) */
    float rewards[N_ACTIONS] = { 2.0f, 5.0f, 
        1.0f, 3.0f };
    /* Action 1 has the highest reward */

    float eps = 0.2f;      /* clipping range */
    int clips = 0;
    float max_ratio = 0;
    float lr = 0.1f;
    int step, i;

    srand(42);

    printf("PPO training (simplified):\n\n");
    printf("  Rewards: [%.1f, %.1f, %.1f, %.1f], "
           "action 1 is best\n\n",
           rewards[0], rewards[1], 
               rewards[2], rewards[3]);

    for (step = 0; step < N_STEPS; step++) {
        /* Save old policy */
        for (i = 0; i < N_ACTIONS; i++)
            old_logits[i] = logits[i];

        float old_probs[N_ACTIONS], 
            new_probs[N_ACTIONS];
        for (i = 0; i < N_ACTIONS; i++)
            old_probs[i] = old_logits[i];
        softmax(old_probs, N_ACTIONS);

        /* Baseline, the reward the old policy
           expects to collect */
        float baseline = 0;
        for (i = 0; i < N_ACTIONS; i++)
            baseline += old_probs[i] * rewards[i];

        /* PPO reuses one batch of data for several
           gradient steps. That reuse is the whole
           reason a ratio exists, since after the
           first inner step the policy has moved and
           the ratio is no longer 1. */
        int inner;
        for (inner = 0; inner < N_INNER; inner++) {
            for (i = 0; i < N_ACTIONS; i++)
                new_probs[i] = logits[i];
            softmax(new_probs, N_ACTIONS);

            for (i = 0; i < N_ACTIONS; i++) {
                float ratio = 
                    new_probs[i]
                        / (old_probs[i] + 1e-8f);
                float adv = rewards[i] - baseline;
                float cl = 
                    clip(ratio, 1.0f - eps, 1.0f + eps);
                float surrogate = 
                    fminf(ratio * adv, cl * adv);

                if (ratio < 1.0f - eps
                    || ratio > 1.0f + eps) clips++;
                if (ratio > max_ratio)
                    max_ratio = ratio;

                logits[i] += 
                    lr * surrogate
                        * (1.0f - new_probs[i]);
            }
        }

        if ((step + 1) % 2 == 0) {
            for (i = 0; i < N_ACTIONS; i++)
                new_probs[i] = logits[i];
            softmax(new_probs, N_ACTIONS);
            printf("  Step %2d: probs=[%.3f, %.3f, "
                   "%.3f, "
                   "%.3f]  max ratio %.2f\n",
                   step + 1, new_probs[0], 
                       new_probs[1], 
                   new_probs[2], 
                       new_probs[3], max_ratio);
            max_ratio = 0;
        }
    }

    printf("\n  The policy moves toward action 1, "
           "which\n");
    printf("  pays the most.\n\n");
    printf("  The clip fired %d times out of %d "
           "checks.\n",
           clips, N_STEPS * N_INNER * N_ACTIONS);
    printf("  Each outer step reuses "
           "one batch for %d\n",
           N_INNER);
    printf("  gradient steps, and the ratio drifts "
           "away\n");
    printf("  from 1 as it does. With a single step "
           "per\n");
    printf("  batch the ratio would be exactly 1 "
           "every\n");
    printf("  time and the clip would never bind.\n");

    return 0;
}
