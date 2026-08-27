/* 131_Causal_Mask.c */
#include <stdio.h>
#include <math.h>
#include <float.h>

#define SEQ 5

static void softmax(float *x, int n)
{
    float mx = -FLT_MAX, s = 0;
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
    /* Raw attention scores (before masking) */
    float scores[SEQ][SEQ];
    int i, j;

    /* Fill with uniform scores */
    for (i = 0; i < SEQ; i++)
        for (j = 0; j < SEQ; j++)
            scores[i][j] = 1.0f;

    printf("Without causal mask (bidirectional):\n\n");
    for (i = 0; i < SEQ; i++) {
        float row[SEQ];
        for (j = 0; j < SEQ; j++) row[j] = scores[i][j];
        softmax(row, SEQ);
        printf("  pos %d attends to: ", i);
        for (j = 0; j < SEQ; j++)
            printf("%.2f ", row[j]);
        printf("\n");
    }

    printf("\nWith causal mask (decoder-style):\n\n");
    for (i = 0; i < SEQ; i++) {
        float row[SEQ];
        for (j = 0; j < SEQ; j++) {
            if (j > i)
                row[j] = -1e9f;
                /* mask future positions */
            else
                row[j] = scores[i][j];
        }
        softmax(row, SEQ);
        printf("  pos %d attends to: ", i);
        for (j = 0; j < SEQ; j++)
            printf("%.2f ", row[j]);
        printf("\n");
    }

    printf("\nPosition 0 can only see itself.\n");
    printf("Position 4 can see all 5 positions.\n");
    printf("This is autoregressive, each position\n");
    printf("generating from what came before it.\n");

    return 0;
}
