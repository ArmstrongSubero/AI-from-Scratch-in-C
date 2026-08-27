/* 106_Projections.c */
#include <stdio.h>

#define D_MODEL 4   /* input dimension */
#define D_K 3        /* key/query dimension */
#define D_V 3        /* value dimension */

/* Matrix multiply: out[i] = sum(W[i][j] * x[j]) */
static void project(const float *x, 
                    const float W[][D_MODEL], 
                     float *out, int out_dim)
{
    int i, j;
    for (i = 0; i < out_dim; i++) {
        out[i] = 0;
        for (j = 0; j < D_MODEL; j++)
            out[i] += W[i][j] * x[j];
    }
}

int main(void)
{
    /* An input vector (e.g., one position's
       embedding) */
    float x[D_MODEL] = { 0.5f, -0.2f, 0.8f, 0.1f };

    /* Three learned projection matrices */
    float W_Q[D_K][D_MODEL] = {
        { 0.3f, -0.1f, 0.4f, 0.2f }, 
        { 0.1f, 0.5f, -0.2f, 0.3f }, 
        { -0.2f, 0.3f, 0.1f, -0.4f }, 
    };
    float W_K[D_K][D_MODEL] = {
        { 0.2f, 0.4f, -0.1f, 0.3f }, 
        { -0.3f, 0.1f, 0.5f, 0.2f }, 
        { 0.4f, -0.2f, 0.3f, -0.1f }, 
    };
    float W_V[D_V][D_MODEL] = {
        { 0.1f, -0.3f, 0.2f, 0.5f }, 
        { 0.4f, 0.2f, -0.1f, 0.3f }, 
        { -0.1f, 0.5f, 0.3f, -0.2f }, 
    };

    float q[D_K], k[D_K], v[D_V];

    project(x, W_Q, q, D_K);
    project(x, W_K, k, D_K);
    project(x, W_V, v, D_V);

    printf("Input x = [%.1f, %.1f, %.1f, %.1f]\n\n",
           x[0], x[1], x[2], x[3]);
    printf("Projected through three different "
           "matrices:\n");
    printf("  Query Q = [%+.3f, %+.3f, %+.3f]\n",
           q[0], q[1], q[2]);
    printf("  Key   K = [%+.3f, %+.3f, %+.3f]\n",
           k[0], k[1], k[2]);
    printf("  Value V = [%+.3f, %+.3f, %+.3f]\n\n",
           v[0], v[1], v[2]);

    printf("Same input, three different "
           "representations.\n");
    printf("Q asks 'what am I looking for?'\n");
    printf("K answers 'what can I be matched on?'\n");
    printf("V holds 'what content do I provide?'\n");

    return 0;
}
