/* 114_Head_Patterns.c */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
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

#define SL 6
#define DM 4
#define NH 2
#define DH (DM / NH)

/* Get attention weights for one head */
static void head_weights(const float X[SL][DM], 
                          const float WQ[DH][DM], 
                          const float WK[DH][DM], 
                          float w[SL][SL])
{
    float scale = 1.0f / sqrtf((float)DH);
    int i, j, k, m;
    for (i = 0; i < SL; i++) {
        float q[DH];
        for (k = 0; k < DH; k++) {
            q[k] = 0;
            for (m = 0; m < DM; m++)
                q[k] += WQ[k][m] * X[i][m];
        }
        float scores[SL];
        for(j = 0;j<SL;j++) {
            float key[DH];
            for (k = 0; k < DH; k++) {
                key[k] = 0;
                for (m = 0; m < DM; m++)
                    key[k] += WK[k][m] * X[j][m];
            }
            scores[j] = dot(q, key, DH)*scale;
            }
        softmax(scores, SL);
        for(j = 0;j<SL;j++) w[i][j] = scores[j];
    }
}

int main(void)
{
    /* A 6-position sequence */
    float X[SL][DM] = {
        /* pos 0: type A */
        { 1.0f, 0.0f, 0.0f, 0.0f }, 
        /* pos 1: type B */
        { 0.0f, 1.0f, 0.0f, 0.0f }, 
        /* pos 2: type A */
        { 1.0f, 0.0f, 0.0f, 0.0f }, 
        /* pos 3: type C */
        { 0.0f, 0.0f, 1.0f, 0.0f }, 
        /* pos 4: type B */
        { 0.0f, 1.0f, 0.0f, 0.0f }, 
        /* pos 5: type D */
        { 0.0f, 0.0f, 0.0f, 1.0f }, 
    };
    const char *labels[] = { "A0", "B1", "A2",
                             "C3", "B4", "D5" };

    /* Hand designed so the two heads differ */

    /* Head 0 attends to the same token type */
    float WQ0[DH][DM] = {{ 1, 0, 0, 0 }, 
        { 0, 1, 0, 0 }};
    float WK0[DH][DM] = {{ 1, 0, 0, 0 }, 
        { 0, 1, 0, 0 }};

    /* Head 1: attends to adjacent positions
       (positional) */
    float WQ1[DH][DM] = {{ 0, 0, 1, 0 }, 
        { 0, 0, 0, 1 }};
    float WK1[DH][DM] = {{ 0, 0, 1, 0 }, 
        { 0, 0, 0, 1 }};

    float w[SL][SL];
    int h, i, j;

    /* Modify X to include positional info in dims
       2-3 */
    for (i = 0; i < SL; i++) {
        X[i][2] = sinf(i * 0.5f);
        X[i][3] = cosf(i * 0.5f);
    }

    printf("Multi-head attention, 2 specialized "
           "heads\n\n");

    /* Head 0 */
    head_weights(X, WQ0, WK0, w);
    printf("Head 0, type matching, A attends to A\n");
    printf("       ");
    for (j = 0; j < SL; j++) printf("%-5s", labels[j]);
    printf("\n");
    for(i = 0;i<SL;i++) {
        printf("  %-4s ",labels[i]);
        for(j = 0;j<SL;j++){
            int bar = (int)(w[i][j] * 20);
            if (bar > 4) printf("####  ");
            else if (bar > 2) printf("##    ");
            else printf(".     ");
        }
        printf("\n");
        }

    /* Head 1 */
    head_weights(X, WQ1, WK1, w);
    printf("\nHead 1, positional, nearby wins\n");
    printf("       ");
    for (j = 0; j < SL; j++) printf("%-5s", labels[j]);
    printf("\n");
    for(i = 0;i<SL;i++) {
        printf("  %-4s ",labels[i]);
        for(j = 0;j<SL;j++){
            int bar = (int)(w[i][j] * 20);
            if (bar > 4) printf("####  ");
            else if (bar > 2) printf("##    ");
            else printf(".     ");
        }
        printf("\n");
        }

    printf("\nHead 0 groups same-type tokens, "
           "A with A.\n");
    printf("Head 1 focuses on positional neighbors.\n");
    printf("Together they capture type and "
           "position.\n");
    printf("One head could not do both "
           "simultaneously.\n");

    return 0;
}
