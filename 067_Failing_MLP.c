/* 067_Failing_MLP.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* The MLP approach, treating the 4-bit sequence
   as 4 independent inputs */

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}
static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define N_HID 8
#define N_IN 4
#define N_PARAMS (N_IN*N_HID + N_HID + N_HID + 1)

static float forward(const float *p, const float *x, 
                     float *h)
{
    int i, j;
    for (i = 0; i < N_HID; i++) {
        float z = p[N_IN * N_HID + i];
        for (j = 0; j < N_IN; j++)
            z += p[i * N_IN + j] * x[j];
        h[i] = sigmoid(z);
    }
    { int bw = N_IN*N_HID + N_HID;
      float z = p[bw + N_HID];
      for (i = 0; i < N_HID; i++)
          z += p[bw + i] * h[i];
      return z; /* linear output for regression */
    }
}

int main(void)
{
    /* Generate all 16 possible 4-bit sequences */
    float X[16][N_IN];
    float T[16];
    int s, b;

    for (s = 0; s < 16; s++) {
        int count = 0;
        for (b = 0; b < N_IN; b++) {
            X[s][b] = (s >> b) & 1 ? 1.0f : 0.0f;
            count += (int)X[s][b];
        }
        T[s] = (float)count;
    }

    printf("The bit-counting problem (length 4):\n\n");
    printf("  Input       Target\n");
    for (s = 0; s < 16; s++) {
        printf("  %.0f%.0f%.0f%.0f      %.0f\n",
               X[s][0], X[s][1], X[s][2], 
                   X[s][3], T[s]);
    }

    printf("\nAn MLP can solve this because ALL "
           "inputs are visible\n");
    printf("at once. It sees the full 4-bit vector "
           "as a fixed input.\n");
    printf("\nBut what if the sequence length "
           "varies? What if it is\n");
    printf("100 bits? 1000 bits? The MLP input "
           "size is fixed.\n");
    printf("You would need a different MLP for "
           "each length.\n");

    printf("\nFor length 4: %d parameters\n", N_PARAMS);
    printf("For length 100: %d parameters "
           "(100*8+8+8+1)\n",
           100 * N_HID + N_HID + N_HID + 1);
    printf("For length 1000: %d parameters\n",
           1000 * N_HID + N_HID + N_HID + 1);

    printf("\nThe parameter count grows linearly "
           "with sequence length.\n");
    printf("And a model trained on length 100 "
           "cannot handle length 101.\n");

    return 0;
}
