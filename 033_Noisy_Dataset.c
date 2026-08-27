/* 033_Noisy_Dataset.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float x[2];
    float t;       /* target: 0 or 1 */
}
Sample;

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

static void generate_data(Sample *data, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        data[i].x[0] = randf() * 4.0f - 2.0f;
        /* range [-2, 2] */
        data[i].x[1] = randf() * 4.0f - 2.0f;

        float dist = data[i].x[0] * data[i].x[0]
                   + data[i].x[1] * data[i].x[1];

        /* Circle with radius 1.2, plus 20% noise */
        float noise = randf() * 0.4f - 0.2f;
        data[i].t = (dist + noise < 1.44f)
            ? 1.0f
            : 0.0f;
    }
}

int main(void)
{
    Sample data[200];
    int i, c0 = 0, c1 = 0;

    srand(42);
    generate_data(data, 200);

    for (i = 0; i < 200; i++) {
        if (data[i].t > 0.5f) c1++;
        else c0++;
    }

    printf("Generated 200 samples\n");
    printf("  Class 0 (outside): %d\n", c0);
    printf("  Class 1 (inside):  %d\n", c1);
    printf("\nFirst 10 samples:\n");
    for (i = 0; i < 10; i++)
        printf("  (%+6.3f, %+6.3f) -> %.0f\n",
               data[i].x[0], data[i].x[1], data[i].t);

    return 0;
}
