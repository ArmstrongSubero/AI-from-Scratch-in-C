/* 163_Grpo.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define GROUP_SIZE 6
#define N_ACTIONS 4

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

/* Sample an action from probability distribution */
static int sample(const float *probs, int n)
{
    float r = randf(), cum = 0;
    int i;
    for (i = 0; i < n; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return n - 1;
}

/* Simulate a reward model: score depends on action */
static float get_reward(int action)
{
    float base_rewards[] = { 2.0f, 5.0f, 1.0f, 3.0f };
    /* Add noise to simulate reward model uncertainty */
    return base_rewards[action]
        + (randf() - 0.5f) * 1.0f;
}

int main(void)
{
    /* uniform start */
    float logits[N_ACTIONS] = { 0, 0, 0, 0 };
    /* the reference policy, which is the SFT model */
    float ref_logits[N_ACTIONS] = { 0, 0, 0, 0 };
    float epsilon = 0.2f;
    float beta = 0.1f;  /* KL penalty weight */
    float lr = 0.3f;
    int step, i, g;

    srand(42);

    printf("GRPO training (DeepSeek-V3 style):\n\n");
    printf("  Group size: %d responses per prompt\n",
           GROUP_SIZE);
    printf("  Rewards: action 1 is best (~5.0)\n\n");

    for (step = 0; step < 20; step++) {
        float probs[N_ACTIONS];
        for (i = 0; i < N_ACTIONS; i++)
            probs[i] = logits[i];
        softmax(probs, N_ACTIONS);

        /* Generate a group of responses */
        int actions[GROUP_SIZE];
        float rewards[GROUP_SIZE];

        for (g = 0; g < GROUP_SIZE; g++) {
            actions[g] = sample(probs, N_ACTIONS);
            rewards[g] = get_reward(actions[g]);
        }

        /* Compute group statistics */
        float mean_r = 0, std_r = 0;
        for (g = 0; g < GROUP_SIZE; g++)
            mean_r += rewards[g];
        mean_r /= GROUP_SIZE;
        for (g = 0; g < GROUP_SIZE; g++) {
            float d = rewards[g] - mean_r;
            std_r += d * d;
        }
        std_r = sqrtf(std_r / GROUP_SIZE + 1e-8f);

        /* GRPO update: advantage is group-relative */
        for (g = 0; g < GROUP_SIZE; g++) {
            float advantage =
                (rewards[g] - mean_r) / std_r;
            int a = actions[g];

            /* Policy gradient with clipping */
            float ref_probs[N_ACTIONS];
            for (i = 0; i < N_ACTIONS; i++)
                ref_probs[i] = ref_logits[i];
            softmax(ref_probs, N_ACTIONS);

            /* Update logit for the chosen action */
            float ratio = probs[a]
                / (ref_probs[a] + 1e-8f);
            float clipped = ratio;
            if (clipped > 1.0f + epsilon)
                clipped = 1.0f + epsilon;
            if (clipped < 1.0f - epsilon)
                clipped = 1.0f - epsilon;

            float obj = fminf(ratio * advantage, 
                              clipped * advantage);

            /* KL penalty: keep close to reference */
            float kl = ref_probs[a] / (probs[a] + 1e-8f)
                      - logf(ref_probs[a]
                        / (probs[a] + 1e-8f)) - 1.0f;

            logits[a] += 
                lr * (obj - beta * kl) / GROUP_SIZE;
        }

        if ((step + 1) % 5 == 0) {
            /* Both numbers must describe the same
               policy, so exp is computed from the
               probabilities the group was drawn
               from rather than from the updated
               ones. Otherwise the columns lag by a
               step and the gap is not just noise. */
            float base[] = { 2.0f, 5.0f, 1.0f, 3.0f };
            float expected = 0;
            for (i = 0; i < N_ACTIONS; i++)
                expected += probs[i] * base[i];

            printf("  Step %2d: probs=[%.2f %.2f %.2f "
                   "%.2f]  exp %.2f  group %.2f\n",
                   step + 1, probs[0], probs[1], 
                   probs[2], probs[3], 
                       expected, mean_r);
        }
    }

    printf("\n  Both columns describe the same "
           "policy.\n");
    printf("  exp is what it earns "
           "on average, worked\n");
    printf("  out exactly. group is an estimate of\n");
    printf("  that same quantity from %d draws, and\n",
           GROUP_SIZE);
    printf("  the two differ by pure "
           "sampling noise.\n");
    printf("  That noise is what the "
           "baseline carries\n");
    printf("  into every advantage, "
           "and it is why the\n");
    printf("  group size is a real choice.\n\n");

    printf("  GRPO advantages over PPO:\n");
    printf("    - No critic, saving a whole model\n");
    printf("    - Group statistics as the baseline\n");
    printf("    - Takes mixed reward "
           "sources easily\n\n");

    printf("  DeepSeek-V3 uses GRPO with:\n");
    printf("    - Rule rewards for math and code\n");
    printf("    - Model rewards for open-ended work\n");
    printf("    - Self-rewarding by voting\n");

    return 0;
}
