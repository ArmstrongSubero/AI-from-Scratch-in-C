/* 112_Concat_Project.c */
#include <stdio.h>
#include <math.h>

#define SEQ_LEN 4
#define D_MODEL 6
#define D_HEAD 3
#define H 2

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


/* One head, returning its output vectors */
static void head(const float X[SEQ_LEN][D_MODEL], 
                 const float W_Q[D_HEAD][D_MODEL], 
                 const float W_K[D_HEAD][D_MODEL], 
                 const float W_V[D_HEAD][D_MODEL], 
                 float out[SEQ_LEN][D_HEAD])
{
    float Q[SEQ_LEN][D_HEAD], K[SEQ_LEN][D_HEAD];
    float V[SEQ_LEN][D_HEAD], w[SEQ_LEN];
    float scale = 1.0f / sqrtf((float)D_HEAD);
    int i, j, k, m;

    for (i = 0; i < SEQ_LEN; i++)
        for (k = 0; k < D_HEAD; k++) {
            Q[i][k] = 0;
            K[i][k] = 0;
            V[i][k] = 0;
            for (m = 0; m < D_MODEL; m++) {
                Q[i][k] += W_Q[k][m] * X[i][m];
                K[i][k] += W_K[k][m] * X[i][m];
                V[i][k] += W_V[k][m] * X[i][m];
            }
        }

    for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < SEQ_LEN; j++) {
            w[j] = 0;
            for (k = 0; k < D_HEAD; k++)
                w[j] += Q[i][k] * K[j][k];
            w[j] *= scale;
        }
        softmax(w, SEQ_LEN);
        for (k = 0; k < D_HEAD; k++) {
            out[i][k] = 0;
            for (j = 0; j < SEQ_LEN; j++)
                out[i][k] += w[j] * V[j][k];
        }
    }
}

int main(void)
{
    float X[SEQ_LEN][D_MODEL] = {
        { 1.0f, 0.2f, -0.3f, 0.5f, -0.1f, 0.8f }, 
        { 0.3f, 0.9f, 0.1f, -0.2f, 0.6f, 0.4f }, 
        { -0.2f, 0.4f, 0.8f, 0.3f, -0.5f, 0.1f }, 
        { 0.6f, -0.1f, 0.2f, 0.9f, 0.3f, -0.4f }, 
    };
    const char *words[SEQ_LEN] = {
        "The", "cat", "sat", "down"
    };

    /* The same two heads as Step 2 */
    float W_Q0[D_HEAD][D_MODEL] = {
        { -0.29f, +0.54f, +1.15f, +0.43f, 
            -0.91f, +0.28f }, 
        { +1.26f, +1.03f, +2.02f, +2.22f, 
            -3.58f, +2.71f }, 
        { +1.40f, +2.66f, +1.56f, +1.52f, 
            +4.53f, -2.21f }, 
    };
    float W_K0[D_HEAD][D_MODEL] = {
        { +0.54f, -0.09f, -0.11f, +0.47f, 
            -0.13f, +0.24f }, 
        { +0.06f, +0.63f, +0.14f, -0.19f, 
            +0.60f, +0.02f }, 
        { -0.01f, +0.28f, +0.76f, +0.55f, 
            -0.26f, -0.15f }, 
    };
    float W_Q1[D_HEAD][D_MODEL] = {
        { +2.02f, +2.52f, -0.38f, -0.18f, 
            +1.01f, +2.57f }, 
        { -0.73f, +1.36f, +2.86f, +1.06f, 
            -2.26f, +0.70f }, 
        { +1.14f, -0.16f, +0.91f, +2.38f, 
            +1.84f, -2.30f }, 
    };
    float W_K1[D_HEAD][D_MODEL] = {
        { +0.57f, -0.09f, -0.09f, +0.52f, 
            -0.09f, +0.19f }, 
        { +0.23f, +0.60f, +0.28f, +0.18f, 
            +0.88f, -0.34f }, 
        { -0.16f, +0.30f, +0.64f, +0.24f, 
            -0.50f, +0.15f }, 
    };
    /* Each head keeps a different slice of content */
    float W_V0[D_HEAD][D_MODEL] = {
        { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, 
        { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f }, 
        { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f }, 
    };
    float W_V1[D_HEAD][D_MODEL] = {
        { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f }, 
        { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f }, 
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f }, 
    };
    /* W_O mixes the two heads back to d_model */
    float W_O[D_MODEL][H * D_HEAD] = {
        { 0.8f, 0.1f, 0.0f, 0.2f, 0.0f, 0.1f }, 
        { 0.1f, 0.7f, 0.1f, 0.0f, 0.3f, 0.0f }, 
        { 0.0f, 0.2f, 0.9f, 0.1f, 0.0f, 0.2f }, 
        { 0.3f, 0.0f, 0.1f, 0.8f, 0.1f, 0.0f }, 
        { 0.0f, 0.2f, 0.0f, 0.1f, 0.7f, 0.2f }, 
        { 0.1f, 0.0f, 0.2f, 0.0f, 0.1f, 0.9f }, 
    };

    float h0[SEQ_LEN][D_HEAD], h1[SEQ_LEN][D_HEAD];
    float cat[SEQ_LEN][H * D_HEAD];
    float out[SEQ_LEN][D_MODEL];
    int i, j, k;

    head(X, W_Q0, W_K0, W_V0, h0);
    head(X, W_Q1, W_K1, W_V1, h1);

    for (i = 0; i < SEQ_LEN; i++) {
        for (k = 0; k < D_HEAD; k++) {
            cat[i][k] = h0[i][k];
            cat[i][D_HEAD + k] = h1[i][k];
        }
        for (j = 0; j < D_MODEL; j++) {
            out[i][j] = 0;
            for (k = 0; k < H * D_HEAD; k++)
                out[i][j] += W_O[j][k] * cat[i][k];
        }
    }

    printf("Per head outputs, "
           "d_head=%d each\n\n", D_HEAD);
    for (i = 0; i < SEQ_LEN; i++) {
        printf("  %-5s h0=[%+.2f,%+.2f,%+.2f]  ",
               words[i], h0[i][0], h0[i][1], h0[i][2]);
        printf("h1=[%+.2f,%+.2f,%+.2f]\n",
               h1[i][0], h1[i][1], h1[i][2]);
    }

    printf("\nConcatenated, "
           "d_model=%d\n\n", H * D_HEAD);
    for (i = 0; i < SEQ_LEN; i++) {
        printf("  %-5s [", words[i]);
        for (k = 0; k < H * D_HEAD; k++)
            printf("%+.2f%s", cat[i][k],
                   k < H*D_HEAD-1 ? "," : "]\n");
    }

    printf("\nAfter the W_O projection, d_model=%d\n\n",
           D_MODEL);
    for (i = 0; i < SEQ_LEN; i++) {
        printf("  %-5s [", words[i]);
        for (j = 0; j < D_MODEL; j++)
            printf("%+.2f%s", out[i][j],
                   j < D_MODEL-1 ? "," : "]\n");
    }

    printf("\nInput and output are both %d wide, "
           "which\n",
           D_MODEL);
    printf("is what lets these layers stack.\n");

    return 0;
}
