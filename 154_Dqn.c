/* 154_Dqn.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}
static float relu(float x)
{
    return x > 0 ? x : 0;
}

#define N_STATES 5
#define N_ACTIONS 2
#define N_HID 16
#define GOAL 4
#define BUFFER_SIZE 200
#define BATCH_SIZE 16

typedef struct {
    float W1[N_HID][N_STATES], b1[N_HID];
    float W2[N_ACTIONS][N_HID], b2[N_ACTIONS];
}
QNet;

static void qnet_init(QNet *q) {
    int i, j;
    for (i = 0; i < N_HID; i++) {
        q->b1[i] = 0;
        for (j = 0; j < N_STATES; j++)
            q->W1[i][j] = (randf()*2-1)*0.3f;
    }
    for (i = 0; i < N_ACTIONS; i++) {
        q->b2[i] = 0;
        for (j = 0; j < N_HID; j++)
            q->W2[i][j] = (randf()*2-1)*0.3f;
    }
}

static void qnet_forward(const QNet *q, int state, 
                         float qv[N_ACTIONS])
{
    float h[N_HID];
    int i, j;
    for (i = 0; i < N_HID; i++)
        h[i] = relu(q->b1[i] + q->W1[i][state]);
    for (i = 0; i < N_ACTIONS; i++) {
        qv[i] = q->b2[i];
        for (j = 0; j < N_HID; j++)
            qv[i] += q->W2[i][j] * h[j];
    }
}

/* Backward pass for one sample */
static void qnet_backward(QNet *q, int state, 
                          int action, float target, 
                          float lr)
{
    float h[N_HID], z[N_HID], qv[N_ACTIONS];
    int i, j;

    /* Forward (save intermediates) */
    for (i = 0; i < N_HID; i++) {
        z[i] = q->b1[i] + q->W1[i][state];
        h[i] = relu(z[i]);
    }
    for (i = 0; i < N_ACTIONS; i++) {
        qv[i] = q->b2[i];
        for (j = 0; j < N_HID; j++)
            qv[i] += q->W2[i][j] * h[j];
    }

    /* Loss gradient, which is
       2 * (qv[action] - target) */
    float d_out = 2.0f * (qv[action] - target);

    /* Output layer gradients */
    for (j = 0; j < N_HID; j++)
        q->W2[action][j] -= lr * d_out * h[j];
    q->b2[action] -= lr * d_out;

    /* Hidden layer gradients */
    for (j = 0; j < N_HID; j++) {
        if (z[j] <= 0) continue;  /* ReLU derivative */
        float dh = d_out * q->W2[action][j];
        q->W1[j][state] -= lr * dh;
        q->b1[j] -= lr * dh;
    }
}

typedef struct {
    int s, a;
    float r;
    int sn, done;
}
Trans;
typedef struct {
    Trans data[BUFFER_SIZE];
    int count, wpos;
}
RB;

static void rb_init(RB *rb)
{
    rb->count = 0;
    rb->wpos = 0;
}
static void rb_add(RB *rb, Trans t) {
    rb->data[rb->wpos] = t;
    rb->wpos = (rb->wpos + 1) % BUFFER_SIZE;
    if (rb->count < BUFFER_SIZE) rb->count++;
    }

int main(void)
{
    QNet policy_net, target_net;
    RB rb;
    float gamma = 0.9f, epsilon, lr = 0.005f;
    int ep, step, i;

    srand(42);
    qnet_init(&policy_net);
    memcpy(&target_net, &policy_net, sizeof(QNet));
    rb_init(&rb);

    printf("DQN Training:\n\n");
    printf("  episode  avg_reward  epsilon  "
           "Q(0,right)  Q(3,right)\n");
    printf("  -------  ----------  -------  "
           "----------  ----------\n");

    float window_sum = 0;
    for (ep = 0; ep < 300; ep++) {
        int s = 0;
        float ep_reward = 0;

        /* Decay epsilon */
        epsilon = 0.3f * (1.0f - (float)ep / 300);
        if (epsilon < 0.05f) epsilon = 0.05f;

        for (step = 0; step < 20; step++) {
            /* Epsilon-greedy action */
            int a;
            if (randf() < epsilon) {
                a = rand() % N_ACTIONS;
            }
            else {
                float qv[N_ACTIONS];
                qnet_forward(&policy_net, s, qv);
                a = qv[1] > qv[0] ? 1 : 0;
            }

            /* Environment step */
            int sn = s;
            if (a == 1 && s < N_STATES-1) sn = s+1;
            else if (a == 0 && s > 0) sn = s-1;
            float r = -1;
            int done = 0;
            if (sn == GOAL) {
                r = 10;
                done = 1;
                }

            /* Store transition */
            Trans t = { s, a, r, sn, done };
            rb_add(&rb, t);
            ep_reward += r;

            /* Train from replay buffer */
            if (rb.count >= BATCH_SIZE) {
                for (i = 0; i < BATCH_SIZE; i++) {
                    Trans sample = 
                        rb.data[rand() % rb.count];
                    float target = sample.r;
                    if (!sample.done) {
                        float qv_next[N_ACTIONS];
                        qnet_forward(&target_net, 
                                     sample.sn, 
                                     qv_next);
                        float best = 
                            qv_next[0] > qv_next[1]
                            ? qv_next[0] : qv_next[1];
                        target += gamma * best;
                    }
                    qnet_backward(&policy_net, 
                                  sample.s, sample.a, 
                                  target, lr);
                }
            }

            s = sn;
            if (done) break;
        }

        /* Update target network periodically */
        if ((ep + 1) % 20 == 0)
            memcpy(&target_net, &policy_net, 
                sizeof(QNet));

        window_sum += ep_reward;

        if ((ep + 1) % 50 == 0) {
            float q0[N_ACTIONS], q3[N_ACTIONS];
            qnet_forward(&policy_net, 0, q0);
            qnet_forward(&policy_net, 3, q3);
            printf("  %5d    %+6.2f      %.2f     "
                   "%+6.2f      %+6.2f\n",
                   ep+1, window_sum / 50.0f, epsilon, 
                   q0[1], q3[1]);
            window_sum = 0;
        }
    }

    /* Final Q-values */
    printf("\nFinal Q-values:\n\n");
    printf("  state   Q(left)   Q(right)  best\n");
    int s;
    for (s = 0; s < N_STATES; s++) {
        float qv[N_ACTIONS];
        qnet_forward(&policy_net, s, qv);
        printf("  %3d     %+6.2f    %+6.2f    %s\n",
               s, qv[0], qv[1], 
               qv[1] > qv[0] ? "right" : "left");
    }

    printf("\nThe DQN reaches the same Q-values the\n");
    printf("table reached in Chapter 29, using a\n");
    printf("network in place of the table.\n");

    return 0;
}
