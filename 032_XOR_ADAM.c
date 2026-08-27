/* 032_XOR_ADAM.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* --- Adam optimizer (from step 6) --- */

typedef struct {
    float *m, *v;
    float beta1, beta2, eps, lr, b1t, b2t;
    int n;
} Adam;

static Adam adam_create(int n, float lr) {
    Adam o;
    o.m = (float*)calloc(n, sizeof(float));
    o.v = (float*)calloc(n, sizeof(float));
    o.beta1 = 0.9f;
    o.beta2 = 0.999f;
    o.eps = 1e-8f;
    o.lr = lr;
    o.b1t = 1.0f;
    o.b2t = 1.0f;
    o.n = n;
    return o;
    }
static void adam_update(Adam *o, float *p, const float *g) {
    int i;
    o->b1t *= o->beta1;
    o->b2t *= o->beta2;
    for (i = 0; i<o->n; i++) { o->m[i] = o->beta1*o->m[i]
        +(1-o->beta1)*g[i]; o->v[i] = o->beta2*o->v[i]
        +(1-o->beta2)*g[i]*g[i];
        float mh = o->m[i]/(1-o->b1t), vh = o->v[i]
            /(1-o->b2t);
        p[i] -= o->lr*mh/(sqrtf(vh)+o->eps);
        } }
static void adam_free(Adam *o)
{
    free(o->m);
    free(o->v);
}

/* --- Network stored as flat arrays --- */

/* Layout: wh[0][0], wh[0][1], wh[1][0],
   wh[1][1], bh[0], bh[1],
           wo[0], wo[1], bo
   Total: 9 parameters */

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

static float forward(const float *p, const float x[2],
    float h[2])
{
    int i, j;
    /* Hidden layer: p[0..3] = weights, p[4..5] = biases */
    for (i = 0; i < 2; i++) {
        float z = p[4 + i];  /* bias */
        for (j = 0; j < 2; j++)
            z += p[i * 2 + j] * x[j];
        h[i] = sigmoid(z);
    }
    /* Output: p[6..7] = weights, p[8] = bias */
    {
        float z = p[8];
        for (i = 0; i < 2; i++)
            z += p[6 + i] * h[i];
        return sigmoid(z);
    }
}

static void backward(const float *p, const float x[2],
    const float h[2],
                      float y, float t, float *g)
{
    int i, j;
    float delta_out = -2.0f * (t - y) * y * (1.0f - y);
    float delta_h[2];

    for (i = 0; i < 2; i++)
        delta_h[i] = delta_out * p[6 + i] * h[i]
            * (1.0f - h[i]);

    /* Zero gradient buffer */
    for (i = 0; i < 9; i++) g[i] = 0.0f;

    /* Output gradients */
    for (i = 0; i < 2; i++)
        g[6 + i] = delta_out * h[i];
    g[8] = delta_out;

    /* Hidden gradients */
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++)
            g[i * 2 + j] = delta_h[i] * x[j];
        g[4 + i] = delta_h[i];
    }
}

int main(void)
{
    float X[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    float T[4] = { 0, 1, 1, 0 };
    float params[9], grads[9];
    Adam opt;
    int epoch, s, i;

    srand(42);
    for (i = 0; i < 9; i++)
        params[i] = ((float)rand() / RAND_MAX) * 2.0f
            - 1.0f;

    opt = adam_create(9, 0.01f);

    for (epoch = 0; epoch < 3000; epoch++) {
        float total_loss = 0.0f;
        float grad_accum[9] = {0};

        for (s = 0; s < 4; s++) {
            float h[2], y, diff;
            y = forward(params, X[s], h);
            diff = T[s] - y;
            total_loss += diff * diff;
            backward(params, X[s], h, y, T[s], grads);
            for (i = 0; i < 9; i++)
                grad_accum[i] += grads[i];
        }

        /* Average gradients over batch */
        for (i = 0; i < 9; i++)
            grad_accum[i] /= 4.0f;

        adam_update(&opt, params, grad_accum);

        if ((epoch + 1) % 500 == 0)
            printf("epoch %4d  loss=%.6f\n", epoch + 1,
                total_loss / 4.0f);
    }

    printf("\nFinal results:\n");
    for (s = 0; s < 4; s++) {
        float h[2];
        float y = forward(params, X[s], h);
        printf("  %.0f XOR %.0f = %.4f  (target %.0f)\n",
               X[s][0], X[s][1], y, T[s]);
    }

    adam_free(&opt);
    return 0;
}
