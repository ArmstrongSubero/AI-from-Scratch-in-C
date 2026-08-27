/* 115_No_Position.c */
#include <stdio.h>
#include <math.h>
#include <float.h>

#define DIM 3
#define SEQ 3

static float dot(const float *a, const float *b, int n)
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

static void run_attention(const char *label, 
                          const float X[SEQ][DIM], 
                           const char *names[SEQ])
{
    float scale = 1.0f / sqrtf((float)DIM);
    int i, j;
    printf("  %s:\n", label);
    for (i = 0; i < SEQ; i++) {
        float scores[SEQ];
        for (j = 0; j < SEQ; j++)
            scores[j] = dot(X[i], X[j], DIM) * scale;
        softmax(scores, SEQ);
        printf("    %-5s attends to: ", names[i]);
        for (j = 0; j < SEQ; j++)
            printf("%s=%.2f ", names[j], scores[j]);
        printf("\n");
    }
}

int main(void)
{
    float A[SEQ][DIM] = {
        { 1.0f, 0.2f, 0.5f },   /* dog */
        { 0.3f, 0.8f, 0.1f },   /* bites */
        { 0.5f, 0.4f, 0.9f },   /* man */
    };
    float B[SEQ][DIM] = {
        /* man (was position 2) */
        { 0.5f, 0.4f, 0.9f }, 
        /* bites (same position) */
        { 0.3f, 0.8f, 0.1f }, 
        /* dog (was position 0) */
        { 1.0f, 0.2f, 0.5f }, 
    };
    const char *names_a[] = { "dog", "bites", "man" };
    const char *names_b[] = { "man", "bites", "dog" };

    printf("Attention without "
           "positional encoding:\n\n");
    run_attention("\"dog bites man\"", A, names_a);
    printf("\n");
    run_attention("\"man bites dog\"", B, names_b);

    printf("\n  Every row is IDENTICAL "
           "across the two\n");
    printf("  orderings. The model cannot tell "
           "subject\n");
    printf("  from object. It needs position.\n");

    return 0;
}
