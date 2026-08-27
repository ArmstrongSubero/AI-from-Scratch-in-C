/* 025_Train_Compare.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

static float sigmoid(float z)
{ return 1.0f / (1.0f + expf(-z)); }

/* 2-input, 4-hidden, 4-output (4-class classification) */
typedef struct {
    float wh[4][2], bh[4];   /* hidden layer */
    float wo[4][4], bo[4];   /* output layer */
} Net;

static void forward(const Net *n, const float x[2],
           float h[4], float logits[4],
           float probs[4])
{
    int i, j;

    /* Hidden layer with sigmoid */
    for (i = 0; i < 4; i++) {
        float z = n->bh[i];
        for (j = 0; j < 2; j++)
            z += n->wh[i][j] * x[j];
        h[i] = sigmoid(z);
    }

    /* Output layer: raw logits */
    for (i = 0; i < 4; i++) {
        logits[i] = n->bo[i];
        for (j = 0; j < 4; j++)
            logits[i] += n->wo[i][j] * h[j];
    }

    /* Softmax */
    {
        float max_val = -FLT_MAX, sum = 0.0f;
        for (i = 0; i < 4; i++)
            if (logits[i] > max_val) max_val = logits[i];
        for (i = 0; i < 4; i++) {
            probs[i] = expf(logits[i] - max_val);
            sum += probs[i];
        }
        for (i = 0; i < 4; i++)
            probs[i] /= sum;
    }
}

static void backward_ce(Net *n,
                        const float x[2], const float h[4],
            const float probs[4],
            int target, float lr)
{
    int i, j;
    float d_out[4], d_h[4];

    /* Output gradient: p - t */
    for (i = 0; i < 4; i++)
        d_out[i] = probs[i];
    d_out[target] -= 1.0f;

    /* Hidden gradient */
    for (i = 0; i < 4; i++) {
        d_h[i] = 0.0f;
        for (j = 0; j < 4; j++)
            d_h[i] += d_out[j] * n->wo[j][i];
        d_h[i] *= h[i] * (1.0f - h[i]);
    }

    /* Update output weights */
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++)
            n->wo[i][j] -= lr * d_out[i] * h[j];
        n->bo[i] -= lr * d_out[i];
    }

    /* Update hidden weights */
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 2; j++)
            n->wh[i][j] -= lr * d_h[i] * x[j];
        n->bh[i] -= lr * d_h[i];
    }
}

static void init_random(Net *n)
{
    int i, j;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 2; j++)
            n->wh[i][j] =
                ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        n->bh[i] = 0.0f;
        for (j = 0; j < 4; j++)
            n->wo[i][j] =
                ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        n->bo[i] = 0.0f;
    }
}

int main(void)
{
    /* 4 quadrants -> 4 classes */
    float X[4][2]  = { {-1,-1}, {-1, 1}, { 1,-1}, { 1, 1} };
    int targets[4] = {  0,       1,       2,       3      };
    Net net;
    float lr = 0.5f;
    int epoch, s;

    srand(42);
    init_random(&net);

    for (epoch = 0; epoch < 5000; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;

        for (s = 0; s < 4; s++) {
            float h[4], logits[4], probs[4];
            float tiny = 1e-9f;

            forward(&net, X[s], h, logits, probs);

            float p = probs[targets[s]];
            if (p < tiny) p = tiny;
            total_loss += -logf(p);

            /* Check accuracy */
            int pred = 0;
            for (int k = 1; k < 4; k++)
                if (probs[k] > probs[pred]) pred = k;
            if (pred == targets[s]) correct++;

            backward_ce(&net, X[s], h, probs,
                        targets[s], lr);
        }

        if ((epoch + 1) % 1000 == 0)
            printf("epoch %4d  loss=%.4f  accuracy=%d/4\n",
                   epoch + 1, total_loss / 4.0f, correct);
    }

    /* Final predictions */
    printf("\nFinal predictions:\n");
    for (s = 0; s < 4; s++) {
        float h[4], logits[4], probs[4];
        forward(&net, X[s], h, logits, probs);
        printf("  x=(%+.0f,%+.0f)  probs="
               "[%.3f %.3f %.3f %.3f]  target=%d\n",
               X[s][0], X[s][1],
               probs[0], probs[1],
               probs[2], probs[3],
               targets[s]);
    }

    return 0;
}
