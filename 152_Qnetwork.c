/* 152_Qnetwork.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define N_STATES 5
#define N_ACTIONS 2
#define N_HID 8

typedef struct {
    float W1[N_HID][N_STATES];  /* input -> hidden */
    float b1[N_HID];
    float W2[N_ACTIONS][N_HID];
    /* hidden -> Q-values */
    float b2[N_ACTIONS];
}
QNet;

static void qnet_init(QNet *q)
{
    int i, j;
    for (i = 0; i < N_HID; i++) {
        q->b1[i] = 0;
        for (j = 0; j < N_STATES; j++)
            q->W1[i][j] = (randf()*2-1) * 0.3f;
    }
    for (i = 0; i < N_ACTIONS; i++) {
        q->b2[i] = 0;
        for (j = 0; j < N_HID; j++)
            q->W2[i][j] = (randf()*2-1) * 0.3f;
    }
}

static float relu(float x)
{
    return x > 0 ? x : 0;
}

static void qnet_forward(const QNet *q, int state, 
                         float qvals[N_ACTIONS])
{
    float h[N_HID];
    int i, j;

    /* One-hot input */
    for (i = 0; i < N_HID; i++) {
        /* one-hot input, so only the column for
           this state contributes anything */
        h[i] = q->b1[i] + q->W1[i][state];
        h[i] = relu(h[i]);
    }
    for (i = 0; i < N_ACTIONS; i++) {
        qvals[i] = q->b2[i];
        for (j = 0; j < N_HID; j++)
            qvals[i] += q->W2[i][j] * h[j];
    }
}

int main(void)
{
    QNet net;
    srand(42);
    qnet_init(&net);

    printf("Q-Network outputs (untrained):\n\n");
    printf("  state   Q(left)   Q(right)  best\n");
    printf("  -----   -------   --------  ----\n");

    int s;
    for (s = 0; s < N_STATES; s++) {
        float qvals[N_ACTIONS];
        qnet_forward(&net, s, qvals);
        printf("  %3d     %+6.3f    %+6.3f    %s\n",
               s, qvals[0], qvals[1], 
               qvals[1] > qvals[0] ? "right" : "left");
    }

    printf("\n  Random weights produce random "
           "Q-values.\n");
    printf("  After training these should match the\n");
    printf("  optimal Q-values from Chapter 29.\n");

    int params = N_HID * N_STATES + N_HID
                 + N_ACTIONS * N_HID + N_ACTIONS;
    printf("\n  Parameters: %d (vs %d Q-table "
           "entries)\n",
           params, N_STATES * N_ACTIONS);

    return 0;
}
