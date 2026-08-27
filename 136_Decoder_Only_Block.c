/* 136_Decoder_Only_Block.c */
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
static void rmsnorm(const float *x, float *out, 
                    int n)
{
    int i;
    double ss = 0;
    float eps = 1e-5f;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    float ri = 1.0f / sqrtf((float)(ss/n) + eps);
    for (i = 0; i < n; i++) out[i] = x[i] * ri;
}

static void init_random(float *p, int n, float scale)
{
    int i;
    for (i = 0; i < n; i++)
        p[i] = (randf() * 2 - 1) * scale;
}

#define SL 5
#define DM 8
#define NH 2
#define DH (DM/NH)
#define DF (DM*4)

typedef struct {
    float WQ[NH][DH][DM], WK[NH][DH][DM];
    float WV[NH][DH][DM], WO[DM][DM];
    float W1[DF][DM], W2[DM][DF];
}
Block;

static void block_forward(const Block *b, 
    float X[SL][DM])
{
    float norm[SL][DM], sub[SL][DM];
    int h, i, j, k, d;

    /* Causal self-attention */
    for (i = 0; i < SL; i++) rmsnorm(X[i], norm[i], DM);

    float ho[NH][SL][DH];
    float scale = 1.0f / sqrtf((float)DH);
    for (h = 0; h < NH; h++)
        for (i = 0; i < SL; i++) {
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
                        key[k] += b->WK[h][k][d]
                                  * norm[j][d];
                }
                sc[j] = dot(q, key, DH) * scale;
                if (j > i) sc[j] = -1e9f;
                /* CAUSAL MASK */
            }
            softmax(sc, SL);
            for (k = 0; k < DH; k++) {
                ho[h][i][k] = 0;
                for (j = 0; j < SL; j++) {
                    float v = 0;
                    for (d = 0; d < DM; d++)
                        v += b
                            ->WV[h][k][d] * norm[j][d];
                    ho[h][i][k] += sc[j] * v;
                }
            }
        }
    for (i = 0; i < SL; i++) {
        float cat[DM];
        for (h = 0; h < NH; h++)
            for (k = 0; k < DH; k++)
                cat[h*DH+k] = ho[h][i][k];
        for (j = 0; j < DM; j++) {
            sub[i][j] = 0;
            for (k = 0; k < DM; k++)
                sub[i][j] += b->WO[j][k] * cat[k];
        }
    }
    for (i = 0; i < SL; i++)
        for (j = 0; j < DM; j++) X[i][j] += sub[i][j];

    /* FFN */
    for (i = 0; i < SL; i++) rmsnorm(X[i], norm[i], DM);
    for (i = 0; i < SL; i++) {
        float hid[DF];
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
            X[i][j] += z;
        }
    }
}

int main(void)
{
    Block blk;
    float X[SL][DM], Y[SL][DM], Z[SL][DM];
    int i, j;

    srand(42);
    init_random((float*)&blk, 
        sizeof(Block)/sizeof(float), 
                0.1f);
    for (i = 0; i < SL; i++)
        for (j = 0; j < DM; j++)
            X[i][j] = randf() * 1.0f - 0.5f;

    /* Run 1, the sequence as given */
    for (i = 0; i < SL; i++)
        for (j = 0; j < DM; j++) Y[i][j] = X[i][j];
    block_forward(&blk, Y);

    /* Run 2, with the LAST token completely replaced */
    for (i = 0; i < SL; i++)
        for (j = 0; j < DM; j++) Z[i][j] = X[i][j];
    for (j = 0; j < DM; j++) Z[SL-1][j] = -Z[SL-1][j];
    block_forward(&blk, Z);

    printf("Causal block, changing only the last "
           "token\n\n");
    printf("  pos   |output|   change from run 1\n");
    printf("  ---   --------   -------------------\n");
    for (i = 0; i < SL; i++) {
        float n = 0, d = 0;
        for (j = 0; j < DM; j++) {
            n += Y[i][j] * Y[i][j];
            d += (Z[i][j] - Y[i][j])
                * (Z[i][j] - Y[i][j]);
        }
        printf("  %3d    %.5f    %.5f%s\n", i,
               sqrtf(n), sqrtf(d), 
               i == SL
                   -1 ? "   <- the changed one" : "");
    }

    printf("\nPositions 0 to %d did not move at all. "
           "The\n",
           SL - 2);
    printf("mask stops them from ever "
           "seeing position\n");
    printf("%d, so replacing it cannot reach them. "
           "Only\n",
           SL - 1);
    printf("the last position changed, "
           "and it changed\n");
    printf("because it is the one that "
           "was edited.\n\n");

    printf("Run the same test on an unmasked block "
           "and\n");
    printf("every row moves. That difference is the\n");
    printf("whole of what causal masking "
           "buys, and it\n");
    printf("is what lets one forward pass produce a\n");
    printf("training signal at every position at "
           "once.\n");

    return 0;
}
