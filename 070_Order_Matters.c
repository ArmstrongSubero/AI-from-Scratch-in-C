/* 070_Order_Matters.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

/* A simple 2-state RNN cell with fixed weights
   chosen to make order effects visible */

static void step(float input, float state[2], 
                 float new_state[2])
{
    new_state[0] = sigmoid(2.0f * input
                           + 1.5f * state[0]
                           - 0.5f * state[1] - 1.0f);
    new_state[1] = sigmoid(-1.0f * input
                           + 0.5f * state[0]
                           + 1.5f * state[1] - 0.5f);
}

static void run_sequence(const float *seq, int len, 
                         const char *label)
{
    float state[2] = { 0, 0 };
    float new_state[2];
    int t;

    printf("  %s: ", label);
    for (t = 0; t < len; t++) {
        step(seq[t], state, new_state);
        state[0] = new_state[0];
        state[1] = new_state[1];
    }
    printf("final state = [%.4f, %.4f]\n",
           state[0], state[1]);
}

int main(void)
{
    float seq_a[] = { 1, 1, 0, 0 };
    float seq_b[] = { 0, 0, 1, 1 };
    float seq_c[] = { 1, 0, 1, 0 };
    float seq_d[] = { 0, 1, 0, 1 };

    printf("Same inputs, different order -> "
           "different final state:\n\n");

    run_sequence(seq_a, 4, "1,1,0,0");
    run_sequence(seq_b, 4, "0,0,1,1");
    run_sequence(seq_c, 4, "1,0,1,0");
    run_sequence(seq_d, 4, "0,1,0,1");

    printf("\nAll four sequences have the same "
           "two 1s and two 0s.\n");
    printf("An MLP treating them as a bag of "
           "inputs would produce\n");
    printf("the same output for all four. The RNN "
           "distinguishes them\n");
    printf("because it processes inputs in order "
           "and the state at\n");
    printf("each step depends on the history.\n");

    return 0;
}
