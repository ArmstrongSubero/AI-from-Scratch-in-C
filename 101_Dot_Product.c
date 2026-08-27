/* 101_Dot_Product.c */
#include <stdio.h>
#include <math.h>

#define DIM 3

/* The label follows the score, so a reader can
   check it rather than take it on trust */
static const char *label(float d)
{
    if (d >  1.2f) return "same direction";
    if (d >  0.8f) return "similar";
    if (d >  0.2f) return "partly aligned";
    if (d > -0.2f) return "orthogonal";
    return "opposite";
}

static float dot(const float a[DIM], const float b[DIM])
{
    float sum = 0;
    int i;
    for (i = 0; i < DIM; i++)
        sum += a[i] * b[i];
    return sum;
}

int main(void)
{
    /* A query vector (what the decoder is looking
       for) */
    float query[DIM] = { 1.0f, 0.0f, 0.5f };

    /* Five key vectors (encoder hidden states) */
    float keys[5][DIM] = {
        /* identical to query */
        { 1.0f, 0.0f, 0.5f }, 
        { 0.8f,  0.1f,  0.4f },   /* similar */
        { 0.0f,  1.0f,  0.0f },   /* orthogonal */
        { -0.5f, 0.2f, -0.3f },   /* opposite-ish */
        { 0.4f,  0.0f,  0.6f },   /* partly aligned */
    };
    int i;

    printf("Dot product as similarity:\n\n");
    printf("  Query: [%.1f, %.1f, %.1f]\n\n",
           query[0], query[1], query[2]);
    printf("  Key                  Dot product  "
           "Interpretation\n");
    printf("  -------------------  -----------  "
           "---------------\n");

    for (i = 0; i < 5; i++) {
        float d = dot(query, keys[i]);
        printf("  [%+4.1f, %+4.1f, %+4.1f]  %+7.2f  "
               "    %s\n",
               keys[i][0], keys[i][1], keys[i][2], 
               d, label(d));
    }

    printf("\nA higher dot product means more similar "
           "and more relevant.\n");
    printf("The decoder pays more attention to "
           "similar states.\n");

    return 0;
}
