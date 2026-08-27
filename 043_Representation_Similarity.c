/* 043_Representation_Similarity.c */
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

#define N_HID 8
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

static float dist(const float *a, const float *b, int n)
{
    float sum = 0;
    int i;
    for (i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

int main(void)
{
    float data_x[200][2], data_t[200];
    float params[N_PARAMS], grads[N_PARAMS], 
        ga[N_PARAMS];
    Adam opt;
    int epoch, s, i;

    srand(42);
    for (i = 0; i < 200; i++) {
        data_x[i][0] = randf() * 4 - 2;
        data_x[i][1] = randf() * 4 - 2;
        float d = data_x[i][0]*data_x[i][0]
            + data_x[i][1]*data_x[i][1];
        data_t[i] = (d < 1.44f) ? 1.0f : 0.0f;
    }
    for (i = 0; i < N_PARAMS; i++)
        params[i] = randf() * 0.4f - 0.2f;
    opt = adam_create(N_PARAMS, 0.01f);

    for (epoch = 0; epoch < 2000; epoch++) {
        for (i = 0; i < N_PARAMS; i++) ga[i] = 0;
        for (s = 0; s < 200; s++) {
            float h[N_HID], y;
            y = forward(params, data_x[s], h);
            backward(params, data_x[s], h, y, 
                data_t[s], grads);
            for (i = 0; i < N_PARAMS; i++)
                ga[i] += grads[i];
        }
        for (i = 0; i < N_PARAMS; i++) ga[i] /= 200.0f;
        adam_update(&opt, params, ga);
    }

    /* Pick a probe point and find nearest/farthest
       in hidden space */
    float probe[2] = { 0.3f, 0.4f };
    float h_probe[N_HID];
    forward(params, probe, h_probe);

    printf("Probe point: (%+.1f, %+.1f)\n",
        probe[0], probe[1]);
    printf("  Input distance vs Hidden distance:\n\n");
    printf("  point       input_dist  hidden_dist  "
           "same_class?\n");

    float test_pts[][2] = {
        { 0.4f,  0.5f},  /* nearby, same class */
        {-0.3f, -0.4f},  /* opposite side, same class */
        /* nearby-ish, different class */
        { 1.8f, 0.0f}, 
        {-1.8f, -1.8f},  /* far, different class */
        { 0.0f,  0.0f},  /* center, same class */
    };
    int n_test = 5;

    for (s = 0; s < n_test; s++) {
        float h_test[N_HID];
        forward(params, test_pts[s], h_test);

        float id = dist(probe, test_pts[s], 2);
        float hd = dist(h_probe, h_test, N_HID);
        float probe_dist = probe[0]*probe[0]
            + probe[1]*probe[1];
        float test_dist = test_pts[s][0]
            *test_pts[s][0] + test_pts[s][1]
            *test_pts[s][1];
        int same = ((probe_dist < 1.44f)
                    == (test_dist < 1.44f));

        printf("  (%+4.1f,%+4.1f)  %8.3f    %8.3f     "
               "%s\n",
               test_pts[s][0], test_pts[s][1], id, hd, 
               same ? "YES" : "NO");
    }

    printf("\nPoints in the same class should have "
           "small hidden distance,\n");
    printf("even if their input distance is large. The "
           "network has learned\n");
    printf("that (-0.3,-0.4) and (0.3,0.4) are 'the "
           "same kind of thing.'\n");

    adam_free(&opt);
    return 0;
}
