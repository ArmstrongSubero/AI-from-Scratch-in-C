/* 088_Reset_Gate.c */
#include <stdio.h>
#include <math.h>

static float my_tanh(float z)
{
    if (z < -20) return -1;
    float e = expf(-2 * z);
    return (1 - e) / (1 + e);
}

#define N 3

int main(void)
{
    float x = 1.0f;
    float h_prev[N] = { 0.8f, -0.5f, 0.3f };
    float W_s[N] = { 0.5f, -0.3f, 0.7f };
    float U_s[N] = { 0.4f, 0.6f, -0.2f };
    float b_s[N] = { 0, 0, 0 };
    int i;

    printf("Reset gate effect on the candidate:\n");
    printf("  s = tanh(W_s * x + U_s * "
           "(r * h_prev) + b_s)\n\n");
    printf("  h_prev = [%.1f, %.1f, %.1f]\n\n",
           h_prev[0], h_prev[1], h_prev[2]);

    float r_values[] = { 0.0f, 0.5f, 1.0f };
    int n_r = 3;

    for (int ri = 0; ri < n_r; ri++) {
        float r = r_values[ri];
        float s[N];

        for (i = 0; i < N; i++) {
            /* reset gate applied to history */
            float rh = r * h_prev[i];
            s[i] = my_tanh(W_s[i] * x
                           + U_s[i] * rh + b_s[i]);
        }

        printf("  r=%.1f: candidate = "
               "[%+.3f, %+.3f, %+.3f]",
               r, s[0], s[1], s[2]);
        if (r < 0.1f) printf("  (ignores history)");
        else if (r > 0.9f) 
            printf("  (uses full history)");
        printf("\n");
    }

    printf("\nWhen r=0, the candidate depends only "
           "on the current input.\n");
    printf("This lets the GRU 'reset' and compute "
           "fresh state from\n");
    printf("scratch, useful when the context "
           "changes abruptly.\n");

    return 0;
}
