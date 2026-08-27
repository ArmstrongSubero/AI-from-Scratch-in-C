/* 104_Seq2seq_Attention.c */
#include <stdio.h>
#include <math.h>

#define ENC_LEN 5
#define DEC_LEN 3
#define DIM 4

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

static float dot(const float *a, const float *b, int n)
{
    float s = 0;
    int i;
    for (i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

int main(void)
{
    const char *in_words[ENC_LEN] = {
        "I", "love", "big", "fat", "cats"
    };
    const char *out_words[DEC_LEN] = {
        "j'aime", "gros", "chats"
    };

    /* Encoder states, one per input word. Chosen rather
       than trained, 
           so the alignment is easy to read. */
    float enc[ENC_LEN][DIM] = {
        { 0.9f,  0.1f,  0.0f,  0.1f },   /* I     */
        { 0.8f,  0.3f,  0.1f,  0.0f },   /* love  */
        { 0.1f,  0.9f,  0.2f,  0.0f },   /* big   */
        { 0.0f,  0.8f,  0.3f,  0.1f },   /* fat   */
        { 0.0f,  0.1f,  0.2f,  0.9f },   /* cats  */
    };

    /* One decoder query per output word */
    float dec[DEC_LEN][DIM] = {
        { 2.5f,  0.0f,  0.0f,  0.0f },   /* j'aime */
        { 0.0f,  2.5f,  0.0f,  0.0f },   /* gros   */
        { 0.0f,  0.0f,  0.0f,  2.5f },   /* chats  */
    };

    float w[ENC_LEN], context[DIM];
    int t, i, j, focus;

    printf("Decoder with attention, "
           "%d output steps\n\n",
           DEC_LEN);

    for (t = 0; t < DEC_LEN; t++) {
        for (i = 0; i < ENC_LEN; i++)
            w[i] = dot(dec[t], enc[i], DIM);
        softmax(w, ENC_LEN);

        for (j = 0; j < DIM; j++) {
            context[j] = 0;
            for (i = 0; i < ENC_LEN; i++)
                context[j] += w[i] * enc[i][j];
        }

        focus = 0;
        for (i = 1; i < ENC_LEN; i++)
            if (w[i] > w[focus]) focus = i;

        printf("  Step %d, generating \"%s\"\n",
               t, out_words[t]);
        printf("    weights: ");
        for (i = 0; i < ENC_LEN; i++)
            printf("%s=%.2f ", in_words[i], w[i]);
        printf("\n    focus:   \"%s\" at %.0f%%\n",
               in_words[focus], w[focus] * 100.0f);
        printf("    context: [%+.3f, %+.3f, %+.3f, "
               "%+.3f]"
               "\n\n",
               context[0], context[1], context[2], 
               context[3]);
    }

    printf("Each step produces its "
           "own distribution,\n");
    printf("so a different context "
           "vector reaches the\n");
    printf("output layer every time. Chapter 18 had "
           "one\n");
    printf("context vector for the whole sequence.\n");

    return 0;
}
