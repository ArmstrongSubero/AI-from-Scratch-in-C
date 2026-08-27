/* 005_Learning.c */
#include <stdio.h>

typedef struct {
    float w[2];
    float b;
} Perceptron;

static float step_function(float z)
{
    return z >= 0.0f ? 1.0f : 0.0f;
}

static float forward(const Perceptron *p, float x0, float x1)
{
    float z = p->w[0] * x0 + p->w[1] * x1 + p->b;
    return step_function(z);
}

int main(void)
{
    /* Training data: AND gate */
    float X[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
    float T[4]    = {  0,     0,     0,     1    };

    Perceptron p = { .w = {0.0f, 0.0f}, .b = 0.0f };
    float lr = 0.1f;
    int epoch, s, i;

    for (epoch = 0; epoch < 20; epoch++) {
        int errors = 0;

        for (s = 0; s < 4; s++) {
            float y   = forward(&p, X[s][0], X[s][1]);
            float err = T[s] - y;

            if (err != 0.0f) {
                errors++;
                for (i = 0; i < 2; i++)
                    p.w[i] += lr * err * X[s][i];
                p.b += lr * err;
            }
        }

        printf("epoch %2d  errors=%d  w0=%.2f w1=%.2f b=%.2f\n",
               epoch + 1, errors, p.w[0], p.w[1], p.b);

        if (errors == 0) {
            printf("Converged!\n");
            break;
        }
    }

    /* Verify */
    printf("\nVerification:\n");
    for (s = 0; s < 4; s++)
        printf("  %.0f AND %.0f = %.0f  (expected %.0f)\n",
               X[s][0], X[s][1],
               forward(&p, X[s][0], X[s][1]), T[s]);

    return 0;
}
