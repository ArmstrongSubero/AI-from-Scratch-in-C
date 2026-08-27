/* 132_Cross_Attention.c */
#include <stdio.h>
#include <math.h>

#define ENC_LEN 5
#define DEC_LEN 3
#define DM 6
#define DK 4

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

int main(void)
{
    /* Encoder outputs, one per source word */
    float enc[ENC_LEN][DM] = {
        { 0.5f, 0.1f, -0.3f, 0.8f, 0.2f, -0.1f }, 
        { 0.3f, 0.7f, 0.1f, -0.2f, 0.5f, 0.4f }, 
        { -0.1f, 0.4f, 0.6f, 0.3f, -0.4f, 0.2f }, 
        { 0.2f, -0.3f, 0.5f, 0.6f, 0.1f, -0.5f }, 
        { 0.8f, 0.2f, -0.1f, 0.4f, 0.3f, 0.6f }, 
    };
    const char *src[ENC_LEN] = {
        "I", "love", "big", "fat", "cats"
    };
    /* Decoder hidden states, one per target word */
    float dec[DEC_LEN][DM] = {
        { 0.4f, 0.6f, -0.2f, 0.3f, 0.1f, 0.5f }, 
        { 0.1f, 0.3f, 0.5f, -0.1f, 0.4f, -0.3f }, 
        { 0.6f, -0.1f, 0.2f, 0.7f, -0.2f, 0.3f }, 
    };
    const char *tgt[DEC_LEN] = { "J'", "aime", "les" };

    /* Chosen rather than trained, so the alignment
       is legible. Training would find its own. */
    float W_Q[DK][DM] = {
        { +0.65f, +3.69f, -1.87f, -0.15f, 
            +1.01f, +2.19f }, 
        { +1.19f, +2.48f, +3.21f, -0.19f, 
            +2.68f, -1.47f }, 
        { +0.53f, -0.21f, +0.76f, +0.54f, 
            +0.05f, -0.20f }, 
        { +2.14f, -1.92f, +2.03f, +2.77f, 
            -0.98f, -0.06f }, 
    };
    float W_K[DK][DM] = {
        { -0.32f, +0.95f, -0.99f, +0.91f, 
            -0.08f, -0.61f }, 
        { -0.08f, +0.94f, +0.26f, -0.26f, 
            +0.94f, -0.46f }, 
        { -0.03f, +0.24f, +1.05f, +0.46f, 
            -0.33f, -0.01f }, 
        { +1.11f, -1.27f, +1.03f, -0.12f, 
            +0.27f, +0.73f }, 
    };
    /* W_V keeps the first four encoder components */
    float W_V[DK][DM] = {
        { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, 
        { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f }, 
        { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f }, 
        { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f }, 
    };

    float scale = 1.0f / sqrtf((float)DK);
    float w[ENC_LEN], ctx[DK];
    int i, j, k, d, best;

    printf("Cross-attention, decoder queries against "
           "encoder keys\n\n");

    for (i = 0; i < DEC_LEN; i++) {
        float q[DK];
        for (k = 0; k < DK; k++) {
            q[k] = 0;
            for (d = 0; d < DM; d++)
                q[k] += W_Q[k][d] * dec[i][d];
        }
        for (j = 0; j < ENC_LEN; j++) {
            float key[DK];
            float sc = 0;
            for (k = 0; k < DK; k++) {
                key[k] = 0;
                for (d = 0; d < DM; d++)
                    key[k] += W_K[k][d] * enc[j][d];
                sc += q[k] * key[k];
            }
            w[j] = sc * scale;
        }
        softmax(w, ENC_LEN);

        for (k = 0; k < DK; k++) {
            ctx[k] = 0;
            for (j = 0; j < ENC_LEN; j++) {
                float v = 0;
                for (d = 0; d < DM; d++)
                    v += W_V[k][d] * enc[j][d];
                ctx[k] += w[j] * v;
            }
        }

        best = 0;
        for (j = 1; j < ENC_LEN; j++)
            if (w[j] > w[best]) best = j;

        printf("  \"%s\" reads the source\n", tgt[i]);
        printf("    weights: ");
        for (j = 0; j < ENC_LEN; j++)
            printf("%s=%.2f ", src[j], w[j]);
        printf("\n    strongest: \"%s\"\n", src[best]);
        printf("    context: [%+.3f, %+.3f, %+.3f, "
               "%+.3f]\n\n",
               ctx[0], ctx[1], ctx[2], ctx[3]);
    }

    printf("Each target word pulls a "
           "different mix of\n");
    printf("source words, and gets a different "
           "context\n");
    printf("vector as a result. Chapter 18 gave the\n");
    printf("decoder one vector for the whole "
           "sentence.\n");

    return 0;
}
