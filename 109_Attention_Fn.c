/* 109_Attention_Fn.c */
#include <stdio.h>
#include <math.h>

#define SL 4    /* sequence length */
#define DM 4    /* model dimension */
#define DK 3    /* query and key dimension */
#define DV 3    /* value dimension */

static void softmax(float *x, int n)
{
    float mx = x[0], s = 0;
    int i;
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    for (i = 0; i < n; i++) x[i] /= s;
}

/* Q/K/V attention over one sequence. Writes the
   weight matrix and the output vectors. */
static void attention(const float X[SL][DM], 
                      const float WQ[DK][DM], 
                      const float WK[DK][DM], 
                      const float WV[DV][DM], 
                      float W[SL][SL], 
                      float out[SL][DV])
{
    float Q[SL][DK], K[SL][DK], V[SL][DV];
    int i, j, k;

    for (i = 0; i < SL; i++) {
        for (k = 0; k < DK; k++) {
            Q[i][k] = 0;
            K[i][k] = 0;
            for (j = 0; j < DM; j++) {
                Q[i][k] += WQ[k][j] * X[i][j];
                K[i][k] += WK[k][j] * X[i][j];
            }
        }
        for (k = 0; k < DV; k++) {
            V[i][k] = 0;
            for (j = 0; j < DM; j++)
                V[i][k] += WV[k][j] * X[i][j];
        }
    }

    for (i = 0; i < SL; i++) {
        for (j = 0; j < SL; j++) {
            W[i][j] = 0;
            for (k = 0; k < DK; k++)
                W[i][j] += Q[i][k] * K[j][k];
        }
        softmax(W[i], SL);

        for (k = 0; k < DV; k++) {
            out[i][k] = 0;
            for (j = 0; j < SL; j++)
                out[i][k] += W[i][j] * V[j][k];
        }
    }
}

int main(void)
{
    float X[SL][DM] = {
        {  1.0f,  0.0f,  0.5f, -0.2f },   /* the  */
        {  0.3f,  0.8f, -0.1f,  0.6f },   /* cat  */
        { -0.2f,  0.1f,  0.9f,  0.3f },   /* sat  */
        {  0.5f, -0.3f,  0.2f,  0.7f },   /* down */
    };
    const char *words[SL] = {
        "the", "cat", "sat", "down"
    };

    /* Same chosen projections as Step 2 */
    float WQ[DK][DM] = {
        { -0.54f, +0.23f, +1.14f, +0.16f }, 
        { +1.01f, +0.59f, +3.72f, -0.67f }, 
        { +1.27f, +0.30f, -1.02f, +3.80f }, 
    };
    float WK[DK][DM] = {
        { +0.92f, -0.40f, +0.21f, +0.11f }, 
        { +0.16f, +0.92f, -0.18f, +0.33f }, 
        { -0.26f, -0.38f, +0.83f, +0.78f }, 
    };
    /* W_V picks out different content from W_K, which
       is the whole point of keeping them separate */
    float WV[DV][DM] = {
        { +1.00f, +0.00f, +0.00f, +0.00f }, 
        { +0.00f, +1.00f, +0.00f, +0.00f }, 
        { +0.00f, +0.00f, +1.00f, +0.00f }, 
    };

    float W[SL][SL], out[SL][DV];
    int i, j;

    attention(X, WQ, WK, WV, W, out);

    printf("Attention weights, rows "
           "query, cols key\n\n");
    printf("          ");
    for (j = 0; j < SL; j++) printf("%-6s", words[j]);
    printf("\n");
    for (i = 0; i < SL; i++) {
        printf("  %-6s  ", words[i]);
        for (j = 0; j < SL; j++)
            printf("%.3f ", W[i][j]);
        printf("\n");
    }

    printf("\nOutput vectors, d_v=%d\n\n", DV);
    for (i = 0; i < SL; i++)
        printf("  %-6s  [%+.3f, %+.3f, %+.3f]\n",
               words[i], out[i][0], 
                   out[i][1], out[i][2]);

    printf("\nEvery row of the weight matrix sums to "
           "1.\n");
    printf("Parameters: W_Q %dx%d + W_K %dx%d + W_V "
           "%dx%d"
           " = %d\n", DK, DM, DK, DM, DV, DM,
           (DK + DK + DV) * DM);

    return 0;
}
