/* 071_Sequence_Modes.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

int main(void)
{
    float seq[] = { 1, 0, 1, 1, 0, 1 };
    int len = 6;

    /* Simulate a counting RNN, hand-tuned
       rather than trained */
    float state = 0.0f;
    int t;

    printf("Sequence-to-sequence (output at "
           "every step):\n\n");
    printf("  step  input  running_count\n");
    for (t = 0; t < len; t++) {
        state += seq[t];
        printf("  %3d     %.0f       %.0f\n", t,
               seq[t], state);
    }

    printf("\nSequence-to-one (output only at "
           "the end):\n");
    printf("  Final count: %.0f\n", state);

    printf("\nSequence-to-sequence tasks:\n");
    printf("  - Language modeling (predict next "
           "word at each position)\n");
    printf("  - Speech recognition (output a "
           "letter per frame)\n");
    printf("  - Part-of-speech tagging (label "
           "each word)\n");

    printf("\nSequence-to-one tasks:\n");
    printf("  - Sentiment analysis (is this "
           "review positive?)\n");
    printf("  - Counting (how many 1s in this "
           "bit stream?)\n");
    printf("  - Classification (what language "
           "is this text?)\n");

    return 0;
}
