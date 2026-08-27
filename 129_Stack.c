/* 129_Stack.c */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}
static float gelu(float x)
{
    float c = 0.7978846f;
    float u = x + 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * u));
}

#define SL 4
#define DM 8
#define NH 2
#define DH (DM/NH)
#define DF (DM*4)

static void rmsnorm(const float *x, float *out, int n) {
    int i;
    double ss = 0;
    float eps = 1e-5f;
    for(i = 0;i<n;i++) ss += x[i]*x[i];
    float ri = 1.0f/sqrtf((float)(ss/n)+eps);
    for(i = 0;i<n;i++) out[i] = x[i]*ri;
    }
static float dot(const float *a, const float *b, 
                 int n)
{
    float s = 0;
    int i;
    for (i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
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

typedef struct {
    float WQ[NH][DH][DM], WK[NH][DH][DM];
    float WV[NH][DH][DM], WO[DM][DM];
    float W1[DF][DM], W2[DM][DF];
}
Block;

static void block_forward(const Block *b, 
    float X[SL][DM]) {
    float norm[SL][DM], sub[SL][DM];
    int h, i, j, k, d;
    /* Attention sublayer */
    for(i = 0;i<SL;i++) rmsnorm(X[i], norm[i], DM);
    float ho[NH][SL][DH];
    float scale = 1.0f/sqrtf((float)DH);
    for(h = 0;h<NH;h++) for(i = 0;i<SL;i++) {
        float q[DH];
        for (k = 0; k < DH; k++) {
            q[k] = 0;
            for (d = 0; d < DM; d++)
                q[k] += b->WQ[h][k][d] * norm[i][d];
        }
        float sc[SL];
        for (j = 0; j < SL; j++) {
            float key[DH];
            for (k = 0; k < DH; k++) {
                key[k] = 0;
                for (d = 0; d < DM; d++)
                    key[k] += b
                        ->WK[h][k][d] * norm[j][d];
            }
            sc[j] = dot(q, key, DH) * scale;
        }
        softmax(sc, SL);
        for (k = 0; k < DH; k++) {
            ho[h][i][k] = 0;
            for (j = 0; j < SL; j++) {
                float v = 0;
                for (d = 0; d < DM; d++)
                    v += b->WV[h][k][d] * norm[j][d];
                ho[h][i][k] += sc[j] * v;
            }
        }
    }
    for (i = 0; i < SL; i++) {
        float cat[DM];
        for (h = 0; h < NH; h++)
            for (k = 0; k < DH; k++)
                cat[h*DH + k] = ho[h][i][k];
        for (j = 0; j < DM; j++) {
            sub[i][j] = 0;
            for (k = 0; k < DM; k++)
                sub[i][j] += b->WO[j][k] * cat[k];
        }
    }
    for(i = 0;i<SL;i++)
        for(j = 0;j<DM;j++)
            X[i][j] += sub[i][j];
    /* FFN sublayer */
    for(i = 0;i<SL;i++) rmsnorm(X[i], norm[i], DM);
    for(i = 0;i<SL;i++) {
        float hid[DF], fo[DM];
        int f;
        for (f = 0; f < DF; f++) {
            float z = 0;
            for (j = 0; j < DM; j++)
                z += b->W1[f][j] * norm[i][j];
            hid[f] = gelu(z);
        }
        for (j = 0; j < DM; j++) {
            float z = 0;
            for (f = 0; f < DF; f++)
                z += b->W2[j][f] * hid[f];
            fo[j] = z;
        }
        for(j = 0;j<DM;j++)X[i][j] += fo[j];
        }
}

int main(void) {
    int N_BLOCKS = 4;
    Block blocks[4];
    float X[SL][DM] = {
        { 1, 0.2f, -0.3f, 0.5f, 
          -0.1f, 0.8f, 0.3f, -0.2f }, 
        {0.3f, 0.9f, 0.1f, -0.2f, 0.6f, 0.4f, 
            -0.1f, 0.5f}, 
        {-0.1f, 0.4f, 0.8f, 0.2f, -0.5f, 0.3f, 
            0.7f, 0.1f}, 
        {0.6f, -0.2f, 0.5f, 0.7f, 0.3f, -0.4f, 
            0.2f, 0.8f}};
    int b, i, j;

    srand(42);
    for (b = 0; b < N_BLOCKS; b++) {
        float *w = (float*)&blocks[b];
        int n = sizeof(Block)/sizeof(float);
        for (i = 0; i < n; i++)
            w[i] = (randf()*2-1)*0.1f;
    }

    printf("Stacking %d transformer "
           "blocks:\n\n", N_BLOCKS);

    /* Compute norm of representations at each depth */
    for (b = 0; b <= N_BLOCKS; b++) {
        float total_norm = 0;
        for (i = 0; i < SL; i++) {
            float n = 0;
            for (j = 0; j < DM; j++)
                n += X[i][j]*X[i][j];
            total_norm += sqrtf(n);
        }
        if (b == 0)
            printf("  input           avg norm = "
                   "%.4f\n", total_norm / SL);
        else
            printf("  after block %d   avg norm = "
                   "%.4f\n", b, total_norm / SL);

        if (b < N_BLOCKS)
            block_forward(&blocks[b], X);
    }

    printf("\nFinal representations:\n");
    const char *words[] = { "The", "cat",
        "sat", "down" };
    for (i = 0; i < SL; i++) {
        printf("  %-5s [", words[i]);
        for (j = 0; j < DM; j++)
            printf("%+.2f%s", X[i][j], j<DM-1?",":"");
        printf("]\n");
    }

    printf("\nThe norms stay controlled because of:\n");
    printf("  1. RMSNorm before each sublayer\n");
    printf("  2. Small random weights at init\n");
    printf("  3. Residuals preserving magnitude\n");

    return 0;
}
