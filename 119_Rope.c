/* 119_Rope.c */
#include <stdio.h>
#include <math.h>

#define DIM 8  /* must be even */

/* Apply RoPE to a vector: rotate pairs of dimensions */
static void apply_rope(const float *x, int dim, 
                       int pos, float *out)
{
    int i;
    for (i = 0; i < dim; i += 2) {
        float freq = 1.0f / powf(10000.0f, 
            (float)i / dim);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        /* 2D rotation of dimensions (i, i+1) */
        out[i] = x[i] * cos_a - x[i + 1] * sin_a;
        out[i + 1] = x[i] * sin_a + x[i + 1] * cos_a;
    }
}

int main(void)
{
    /* A query and a key vector */
    float q[DIM] = { 1.0f, 0.0f, 0.5f, 0.5f, 
                     0.3f, 0.7f, 0.2f, 0.8f };
    float k[DIM] = { 0.8f, 0.2f, 0.6f, 0.4f, 
                     0.5f, 0.5f, 0.1f, 0.9f };
    int i;

    printf("Rotary Position Embeddings (RoPE):\n\n");
    printf("Original query: [");
    for (i = 0; i < DIM; i++)
        printf("%.1f%s", q[i], i<DIM-1?", ":"");
    printf("]\n\n");

    /* Show how the query changes with position */
    printf("Query rotated at different positions:\n");
    int positions[] = { 0, 1, 2, 5, 10 };
    int n_pos = 5;

    for (int p = 0; p < n_pos; p++) {
        float q_rot[DIM];
        apply_rope(q, DIM, positions[p], q_rot);
        printf("  pos %2d: [", positions[p]);
        for (i = 0; i < DIM; i++)
            printf("%+.2f%s", q_rot[i],
                i<DIM-1?", ":"");
        printf("]\n");
    }

    /* Show that dot product encodes relative
       position */
    printf("\nDot product of q(pos_q) and k(pos_k)\n");
    printf("  relative distance matters, not the "
           "absolute position\n\n");
    printf("  pos_q  pos_k  dist  dot_product\n");

    int test_pairs[][2] = { {0, 0}, {0, 1}, {0, 2}, 
                            {5, 6}, {5, 7}, {10, 11} };
    int n_tests = 6;

    for (int t = 0; t < n_tests; t++) {
        int pq = test_pairs[t][0], 
            pk = test_pairs[t][1];
        float q_rot[DIM], k_rot[DIM];
        apply_rope(q, DIM, pq, q_rot);
        apply_rope(k, DIM, pk, k_rot);

        float dp = 0;
        for (i = 0; i < DIM; i++)
            dp += q_rot[i] * k_rot[i];

        printf("  %3d    %3d    %3d   %+.4f\n",
               pq, pk, pk - pq, dp);
    }

    printf("\nPairs the same distance apart, 0-1 "
           "and 5-6 and 10-11,\n");
    printf("give the same dot product wherever "
           "they sit.\n");
    printf("RoPE puts RELATIVE position straight "
           "into the\n");
    printf("attention score through rotation.\n");

    return 0;
}
