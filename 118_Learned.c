/* 118_Learned.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define D_MODEL 4
#define MAX_POS 8

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

/* Learned table, one row per position. Untrained
   here, so the values are only placeholders. */
static float table[MAX_POS][D_MODEL];

/* Returns 0 and leaves out untouched when the
   position is past the end of the table */
static int learned_pe(int pos, float out[D_MODEL])
{
    int i;
    if (pos < 0 || pos >= MAX_POS) return 0;
    for (i = 0; i < D_MODEL; i++)
        out[i] = table[pos][i];
    return 1;
}

/* Sinusoidal needs no table, so it is defined
   for every position there is */
static void sinusoidal_pe(int pos, float out[D_MODEL])
{
    int i;
    for (i = 0; i < D_MODEL; i++) {
        float e = (float)(i / 2 * 2) / D_MODEL;
        float angle = pos / powf(10000.0f, e);
        out[i] = (i % 2 == 0)
            ? sinf(angle)
            : cosf(angle);
    }
}

int main(void)
{
    float v[D_MODEL];
    int pos, i;

    srand(42);
    for (pos = 0; pos < MAX_POS; pos++)
        for (i = 0; i < D_MODEL; i++)
            table[pos][i] = randf() * 0.4f - 0.2f;

    printf("Learned table, d_model=%d, max_pos=%d\n\n",
           D_MODEL, MAX_POS);
    for (pos = 0; pos < MAX_POS; pos++) {
        learned_pe(pos, v);
        printf("  pos %d: [%+.3f, %+.3f, %+.3f, "
               "%+.3f]\n",
               pos, v[0], v[1], v[2], v[3]);
    }

    printf("\nAsking for positions past the table\n\n");
    printf("  pos   learned          sinusoidal\n");
    printf("  ---   --------------   ----------\n");
    for (pos = 6; pos <= 10; pos++) {
        float s[D_MODEL];
        int ok = learned_pe(pos, v);
        sinusoidal_pe(pos, s);
        printf("  %2d    ", pos);
        if (ok) printf("%+.3f %+.3f    ", v[0], v[1]);
        else    printf("no entry         ");
        printf("%+.3f %+.3f\n", s[0], s[1]);
    }

    printf("\nThe table stops at %d. "
           "Sinusoidal keeps\n",
           MAX_POS - 1);
    printf("going because it is a "
           "formula rather than\n");
    printf("a lookup, defined at every integer.\n\n");

    printf("Table cost: %d x %d = %d floats\n",
           MAX_POS, D_MODEL, MAX_POS * D_MODEL);
    printf("At max_pos=2048, d_model=768 "
           "that becomes\n");
    printf("%d floats, or %d KB at 4 bytes each.\n",
           2048 * 768, 2048 * 768 * 4 / 1024);

    return 0;
}
