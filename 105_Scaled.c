/* 105_Scaled.c */
#include <stdio.h>
#include <math.h>
#include <float.h>

#define N_KEYS 4

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

static void show_weights(const char *label, 
                         const float *scores, int n)
{
    float w[N_KEYS];
    int i;
    for (i = 0; i < n; i++) w[i] = scores[i];
    softmax(w, n);
    printf("  %-12s scores=[%+5.1f, %+5.1f, "
           "%+5.1f, %+5.1f]  "
           "weights=[%.3f, %.3f, %.3f, %.3f]\n",
           label, scores[0], scores[1], 
           scores[2], scores[3], 
           w[0], w[1], w[2], w[3]);
}

int main(void)
{
    printf("Effect of scaling on "
           "attention weights:\n\n");

    /* Small dimension (d=4): scores are small */
    float small_scores[N_KEYS] = 
        { 1.2f, 0.8f, 0.3f, -0.5f };
    show_weights("d=4", small_scores, N_KEYS);

    /* Large dimension (d=512): scores are much
       larger */
    float large_scores[N_KEYS] = 
        { 15.0f, 10.0f, 3.0f, -6.0f };
    show_weights("d=512 (raw)", large_scores, N_KEYS);

    /* After scaling by 1/sqrt(512) = 1/22.6 */
    float scale = 1.0f / sqrtf(512.0f);
    float scaled[N_KEYS];
    int i;
    for (i = 0; i < N_KEYS; i++)
        scaled[i] = large_scores[i] * scale;
    show_weights("d=512 (scaled)", scaled, N_KEYS);

    printf("\nWithout scaling, large-d scores push "
           "softmax into\n");
    printf("near-one-hot mode. Only one key gets "
           "all the weight.\n");
    printf("Scaling by 1/sqrt(d) keeps the "
           "distribution soft,\n");
    printf("allowing gradients to flow to all keys.\n");

    printf("\nThis is the 'scaled' in 'scaled "
           "dot-product attention'.\n");
    printf("Every transformer uses it.\n");

    return 0;
}
