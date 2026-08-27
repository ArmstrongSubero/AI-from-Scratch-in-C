/* 041_Watching_Representations_Form.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}
static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define N_HID 2  /* 2 hidden so we can think in 2D */
#define N_PARAMS (2*N_HID + N_HID + N_HID + 1)

static float forward(const float *p, const float x[2], 
    float h[N_HID])
{
    int i, j;
    for (i = 0; i < N_HID; i++) {
        float z = p[2*N_HID + i];
        for (j = 0; j < 2; j++) z += p[i*2+j] * x[j];
        h[i] = sigmoid(z);
    }
    { int bw = 3*N_HID; float z = p[bw+N_HID];
      for (i = 0; i < N_HID; i++) z += p[bw+i] * h[i];
      return sigmoid(z);
      }
}

static void backward(const float *p, const float x[2], 
    const float h[N_HID], 
                      float y, float t, float *g)
{
    int i, j, bh = 2*N_HID, bw = 3*N_HID;
    float d_out = -2.0f * (t - y) * y * (1.0f - y);
    for (i = 0; i < N_PARAMS; i++) g[i] = 0;
    for (i = 0; i < N_HID; i++) g[bw+i] = d_out * h[i];
    g[bw+N_HID] = d_out;
    for (i = 0; i < N_HID; i++) {
        float dh = d_out * p[bw+i] * h[i]
            * (1.0f - h[i]);
        for (j = 0; j < 2; j++) g[i*2+j] = dh * x[j];
        g[bh+i] = dh;
    }
}

typedef struct { float *m, *v; float b1, b2, eps, lr, 
    b1t, b2t;
    int n;
    }
    Adam;
static Adam adam_create(int n, float lr) {
    Adam o;
    o.m = (float*)calloc(n, sizeof(float));
    o.v = (float*)calloc(n, sizeof(float));
    o.b1 = 0.9f;
    o.b2 = 0.999f;
    o.eps = 1e-8f;
    o.lr = lr;
    o.b1t = 1;
    o.b2t = 1;
    o.n = n;
    return o;
    }
static void adam_update(Adam *o, float *p, 
    const float *g) {
    int i;
    o->b1t *= o->b1;
    o->b2t *= o->b2;
    for(i = 0;i<o->n;i++) {
        o->m[i] = o->b1*o->m[i]+(1-o->b1)*g[i];
        o->v[i] = o->b2*o->v[i]+(1-o->b2)*g[i]*g[i];
        float mh = o->m[i]/(1-o->b1t), 
            vh = o->v[i]/(1-o->b2t);
        p[i] -= o->lr*mh/(sqrtf(vh)+o->eps);
        }
        }
static void adam_free(Adam *o)
{
    free(o->m);
    free(o->v);
}

int main(void)
{
    /* XOR so we can use 2 hidden neurons */
    float X[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    float T[4] = { 0, 1, 1, 0 };
    float params[N_PARAMS], grads[N_PARAMS], 
        ga[N_PARAMS];
    Adam opt;
    int epoch, s, i;
    int snapshots[] = { 0, 10, 50, 200, 1000, 5000 };
    int n_snaps = 6;
    int snap_idx = 0;

    srand(42);
    for (i = 0; i < N_PARAMS; i++)
        params[i] = randf() * 2 - 1;
    opt = adam_create(N_PARAMS, 0.01f);

    printf("How hidden representations form during "
           "training (2 hidden neurons):\n\n");

    for (epoch = 0; epoch <= 5000; epoch++) {
        /* Print snapshot */
        if (snap_idx < n_snaps
                && epoch == snapshots[snap_idx]) {
            printf("Epoch %d:\n", epoch);
            printf("  input     h0      h1      "
                   "output\n");
            for (s = 0; s < 4; s++) {
                float h[N_HID], y;
                y = forward(params, X[s], h);
                printf("  (%.0f, %.0f)   %6.4f  %6.4f  "
                       "%6.4f  (target %.0f)\n",
                       X[s][0], X[s][1], h[0], 
                           h[1], y, T[s]);
            }
            printf("\n");
            snap_idx++;
        }

        /* Train one epoch */
        for (i = 0; i < N_PARAMS; i++) ga[i] = 0;
        for (s = 0; s < 4; s++) {
            float h[N_HID], y;
            y = forward(params, X[s], h);
            backward(params, X[s], h, y, T[s], grads);
            for (i = 0; i < N_PARAMS; i++)
                ga[i] += grads[i];
        }
        for (i = 0; i < N_PARAMS; i++) ga[i] /= 4.0f;
        adam_update(&opt, params, ga);
    }

    adam_free(&opt);
    return 0;
}
