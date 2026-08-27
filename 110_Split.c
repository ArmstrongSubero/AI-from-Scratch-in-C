/* 110_Split.c */
#include <stdio.h>

#define D_MODEL 8
#define H 2
#define D_HEAD (D_MODEL / H)

/* The naive picture, slice the vector into h pieces */
static void slice(const float x[D_MODEL], int head, 
                  float out[D_HEAD])
{
    int i;
    for (i = 0; i < D_HEAD; i++)
        out[i] = x[head * D_HEAD + i];
}

/* What actually happens, project the FULL vector
   down to d_head with this head's own matrix */
static void project(const float x[D_MODEL], 
                    const float W[D_HEAD][D_MODEL], 
                    float out[D_HEAD])
{
    int i, j;
    for (i = 0; i < D_HEAD; i++) {
        out[i] = 0;
        for (j = 0; j < D_MODEL; j++)
            out[i] += W[i][j] * x[j];
    }
}

int main(void)
{
    float x[D_MODEL] = { 0.1f, 0.5f, -0.3f, 0.8f, 
                        -0.2f, 0.4f, 0.7f, -0.1f };

    /* Head 0 reads mostly the front of the vector,
       head 1 mostly the back, but both see all of it */
    float W0[D_HEAD][D_MODEL] = {
        { 0.9f, 0.1f, 0.0f, 0.0f, 
          0.2f, 0.0f, 0.0f, 0.0f }, 
        { 0.0f, 0.8f, 0.2f, 0.0f, 
          0.0f, 0.1f, 0.0f, 0.0f }, 
        { 0.0f, 0.0f, 0.7f, 0.3f, 
          0.0f, 0.0f, 0.1f, 0.0f }, 
        { 0.1f, 0.0f, 0.0f, 0.9f, 
          0.0f, 0.0f, 0.0f, 0.2f }, 
    };
    float W1[D_HEAD][D_MODEL] = {
        { 0.2f, 0.0f, 0.0f, 0.0f, 
          0.9f, 0.1f, 0.0f, 0.0f }, 
        { 0.0f, 0.1f, 0.0f, 0.0f, 
          0.0f, 0.8f, 0.2f, 0.0f }, 
        { 0.0f, 0.0f, 0.1f, 0.0f, 
          0.0f, 0.0f, 0.9f, 0.3f }, 
        { 0.0f, 0.0f, 0.0f, 0.2f, 
          0.1f, 0.0f, 0.0f, 0.8f }, 
    };

    float s0[D_HEAD], s1[D_HEAD];
    float p0[D_HEAD], p1[D_HEAD];
    int i;

    slice(x, 0, s0);
    slice(x, 1, s1);
    project(x, W0, p0);
    project(x, W1, p1);

    printf("d_model=%d, %d heads, d_head=%d\n\n",
           D_MODEL, H, D_HEAD);
    printf("Full vector:  [");
    for (i = 0; i < D_MODEL; i++)
        printf("%+.1f%s", x[i],
               i < D_MODEL-1 ? ", " : "]\n\n");

    printf("The naive picture, slicing the vector\n");
    printf("  head 0:  [");
    for (i = 0; i < D_HEAD; i++)
        printf("%+.2f%s", s0[i],
               i < D_HEAD-1 ? ", " : "]\n");
    printf("  head 1:  [");
    for (i = 0; i < D_HEAD; i++)
        printf("%+.2f%s", s1[i],
               i < D_HEAD-1 ? ", " : "]\n\n");

    printf("What really happens, each head projects\n");
    printf("the whole vector through its own matrix\n");
    printf("  head 0:  [");
    for (i = 0; i < D_HEAD; i++)
        printf("%+.2f%s", p0[i],
               i < D_HEAD-1 ? ", " : "]\n");
    printf("  head 1:  [");
    for (i = 0; i < D_HEAD; i++)
        printf("%+.2f%s", p1[i],
               i < D_HEAD-1 ? ", " : "]\n\n");

    printf("The projected values differ from the "
           "slice\n");
    printf("because every head can read every input\n");
    printf("component. What is split is the output\n");
    printf("width, not the input.\n");

    return 0;
}
