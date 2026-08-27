/* 040_Circle_Hidden.c */
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

#define N_HID 4  /* small enough to inspect */
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

typedef struct { float x[2]; float t; } Sample;

int main(void)
{
    Sample data[200];
    float params[N_PARAMS], grads[N_PARAMS], 
        ga[N_PARAMS];
    Adam opt;
    int epoch, s, i;

    srand(42);
    /* Generate circle data */
    for (i = 0; i < 200; i++) {
        data[i].x[0] = randf() * 4 - 2;
        data[i].x[1] = randf() * 4 - 2;
        float dist = data[i].x[0]*data[i].x[0]
            + data[i].x[1]*data[i].x[1];
        data[i].t = (dist < 1.44f) ? 1.0f : 0.0f;
        /* no noise for clarity */
    }
    for (i = 0; i < N_PARAMS; i++)
        params[i] = randf() * 0.4f - 0.2f;
    opt = adam_create(N_PARAMS, 0.01f);

    /* Train */
    for (epoch = 0; epoch < 1000; epoch++) {
        for (i = 0; i < N_PARAMS; i++) ga[i] = 0;
        for (s = 0; s < 200; s++) {
            float h[N_HID], y;
            y = forward(params, data[s].x, h);
            backward(params, data[s].x, h, y, 
                data[s].t, grads);
            for (i = 0; i < N_PARAMS; i++)
                ga[i] += grads[i];
        }
        for (i = 0; i < N_PARAMS; i++) ga[i] /= 200.0f;
        adam_update(&opt, params, ga);
    }

    /* Inspect: what does each hidden neuron respond
       to? */
    printf("Hidden neuron weights (what each neuron "
           "detects):\n\n");
    for (i = 0; i < N_HID; i++) {
        printf("  Neuron %d: w0=%+.3f  w1=%+.3f  "
               "bias=%+.3f\n",
               i, params[i*2], params[i*2+1], 
                   params[2*N_HID+i]);
    }

    /* Sample a few points and show their hidden
       representations */
    printf("\nHidden representations for selected "
           "points:\n\n");
    printf("  x0      x1     dist  ");
    for (i = 0; i < N_HID; i++) printf("  h%d    ", i);
    printf("  output  class\n");

    float test_points[][2] = {
        { 0.0f,  0.0f},  /* center */
        { 0.5f,  0.5f},  /* inside */
        {-0.8f,  0.3f},  /* inside */
        { 1.0f,  0.0f},  /* on boundary */
        { 1.5f,  0.0f},  /* outside */
        { 0.0f,  1.5f},  /* outside */
        {-1.5f, -1.5f},  /* far outside */
    };
    int n_test = 7;

    for (s = 0; s < n_test; s++) {
        float h[N_HID], y;
        float dist = test_points[s][0]*test_points[s][0]
                   + test_points[s][1]
                       *test_points[s][1];
        y = forward(params, test_points[s], h);
        printf("  %+5.1f   %+5.1f  %5.2f",
            test_points[s][0], test_points[s][1], dist);
        for (i = 0; i < N_HID; i++)
            printf("  %5.3f", h[i]);
        printf("  %5.3f   %s\n", y,
            dist < 1.44f ? "IN" : "OUT");
    }

    adam_free(&opt);
    return 0;
}
