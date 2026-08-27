/* 069_Learned_State.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}
static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

/* A simple recurrent cell:
   new_state = sigmoid(W_x * input
                       + W_h * old_state + bias) */

#define STATE_SIZE 4

typedef struct {
    float w_x[STATE_SIZE];      /* input weight */
    /* state-to-state weight */
    float w_h[STATE_SIZE][STATE_SIZE];
    float b[STATE_SIZE];        /* bias */
    float w_out[STATE_SIZE];    /* output weight */
    float b_out;                /* output bias */
}
RNNCell;

static void rnn_step(const RNNCell *cell, float input, 
                      const float state_in[STATE_SIZE], 
                      float state_out[STATE_SIZE])
{
    int i, j;
    for (i = 0; i < STATE_SIZE; i++) {
        float z = cell->b[i] + cell->w_x[i] * input;
        for (j = 0; j < STATE_SIZE; j++)
            z += cell->w_h[i][j] * state_in[j];
        state_out[i] = sigmoid(z);
    }
}

static float rnn_output(const RNNCell *cell, 
                         const float state[STATE_SIZE])
{
    float z = cell->b_out;
    int i;
    for (i = 0; i < STATE_SIZE; i++)
        z += cell->w_out[i] * state[i];
    return z;  /* linear output */
}

int main(void)
{
    RNNCell cell;
    float state[STATE_SIZE];
    int i, j;

    srand(42);

    /* Random initialization */
    for (i = 0; i < STATE_SIZE; i++) {
        cell.w_x[i] = randf() * 2 - 1;
        cell.b[i] = 0;
        cell.w_out[i] = randf() * 2 - 1;
        for (j = 0; j < STATE_SIZE; j++)
            cell.w_h[i][j] = randf() * 2 - 1;
    }
    cell.b_out = 0;

    /* Process a sequence */
    float sequence[] = { 1, 0, 1, 1, 0, 1, 0, 1 };
    int length = 8;

    /* Initialize state to zeros */
    for (i = 0; i < STATE_SIZE; i++)
        state[i] = 0.0f;

    printf("Processing sequence through untrained "
           "RNN cell:\n\n");
    printf("  step  input  state                "
           "        output\n");
    for (int t = 0; t < length; t++) {
        float new_state[STATE_SIZE];
        rnn_step(&cell, sequence[t], state, new_state);

        float out = rnn_output(&cell, new_state);

        printf("  %3d     %.0f    [", t, sequence[t]);
        for (i = 0; i < STATE_SIZE; i++)
            printf("%.3f%s", new_state[i],
                   i < STATE_SIZE - 1 ? ", " : "");
        printf("]  %.3f\n", out);

        /* Copy new state for next step */
        for (i = 0; i < STATE_SIZE; i++)
            state[i] = new_state[i];
    }

    printf("\nThe state vector changes at every "
           "step.\n");
    printf("It encodes information about all "
           "past inputs.\n");
    printf("With training, the output would learn "
           "to count 1s.\n");
    printf("The SAME cell is used at every time "
           "step.\n");
    printf("\nParameters: %d (independent of "
           "sequence length)\n",
           STATE_SIZE + STATE_SIZE*STATE_SIZE
               + STATE_SIZE +
           STATE_SIZE + 1);

    return 0;
}
