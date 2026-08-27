/* 161_Reward_Model.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

#define FEAT_DIM 4

/* Simple reward model: linear function of features */
typedef struct {
    float w[FEAT_DIM];
    float bias;
}
RewardModel;

static float reward_score(const RewardModel *rm, 
                          const float f[FEAT_DIM])
{
    float score = rm->bias;
    int i;
    for (i = 0; i < FEAT_DIM; i++)
        score += rm->w[i] * f[i];
    return score;
}

/* Train on preference pairs using the Bradley-Terry
   model:
   P(response_a > response_b) =
       sigmoid(score_a - score_b)
   Loss = 
       -log(sigmoid(score_chosen - score_rejected)) */

static float train_step(RewardModel *rm, 
                         const float chosen[FEAT_DIM], 
            const float rejected[FEAT_DIM], 
                         float lr)
{
    float s_c = reward_score(rm, chosen);
    float s_r = reward_score(rm, rejected);
    float diff = s_c - s_r;
    float prob = sigmoid(diff);
    float loss = -logf(prob + 1e-8f);

    /* d_loss/d_diff = -(1 - prob) = prob - 1 */
    float grad = prob - 1.0f;
    int i;
    for (i = 0; i < FEAT_DIM; i++) {
        rm->w[i] -= lr * grad
            * (chosen[i] - rejected[i]);
    }

    /* The bias gets no update, and that is not an
       oversight. It appears in both scores, so it
       cancels out of the difference the loss is
       built from, and its gradient is exactly
       1 - 1 = 0. Pairwise data cannot pin down an
       absolute offset, only relative ones. */

    return loss;
}

int main(void)
{
    RewardModel rm = { .w = {0}, .bias = 0 };

    /* Preference data: features represent response
       quality
       [helpfulness, accuracy, conciseness, safety] */
    float pairs[][2][FEAT_DIM] = {
        /* chosen,                     rejected */
        {{ 0.9f, 0.8f, 0.7f, 1.0f }, 
         { 0.3f, 0.2f, 0.5f, 0.8f }}, 
        {{ 0.7f, 0.9f, 0.6f, 1.0f }, 
         { 0.8f, 0.1f, 0.9f, 0.3f }}, 
        {{ 0.8f, 0.7f, 0.8f, 0.9f }, 
         { 0.2f, 0.6f, 0.3f, 1.0f }}, 
        {{ 0.6f, 0.8f, 0.5f, 1.0f }, 
         { 0.9f, 0.7f, 0.8f, 0.1f }}, 
        {{ 0.8f, 0.9f, 0.7f, 0.9f }, 
         { 0.4f, 0.3f, 0.6f, 0.7f }}, 
    };
    int n_pairs = 5;

    printf("Training a reward model on preference "
           "pairs\n\n");

    int epoch, p;
    for (epoch = 0; epoch < 200; epoch++) {
        float total_loss = 0;
        for (p = 0; p < n_pairs; p++)
            total_loss += train_step(&rm, pairs[p][0], 
                                     pairs[p][1], 0.1f);

        if ((epoch + 1) % 50 == 0) {
            /* Check accuracy */
            int correct = 0;
            for (p = 0; p < n_pairs; p++) {
                float sc = reward_score(&rm, 
                    pairs[p][0]);
                float sr = reward_score(&rm, 
                    pairs[p][1]);
                if (sc > sr) correct++;
            }
            printf("  Epoch %3d: loss=%.3f  "
                   "accuracy=%d/%d\n",
                   epoch + 1, total_loss / n_pairs, 
                   correct, n_pairs);
        }
    }

    printf("\nLearned weights:\n");
    const char *names[] = { "helpfulness", "accuracy",
                            "conciseness", "safety" };
    for (int i = 0; i < FEAT_DIM; i++)
        printf("  %-14s  %+.3f\n", names[i], rm.w[i]);
    printf("  bias            %+.3f  <- cannot move\n",
           rm.bias);

    printf("\nThe bias stays where it started. It "
           "sits\n");
    printf("in both scores and cancels in the\n");
    printf("difference, so preference data cannot\n");
    printf("say anything about it at all.\n\n");
    printf("The weights are what humans valued.\n");
    printf("Safety scores high because humans kept\n");
    printf("preferring the safe response.\n");

    return 0;
}
