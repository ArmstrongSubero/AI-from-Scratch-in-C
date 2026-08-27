/* 153_Replay.c */
#include <stdio.h>
#include <stdlib.h>

#define BUFFER_SIZE 100
#define BATCH_SIZE 8

typedef struct {
    int state;
    int action;
    float reward;
    int next_state;
    int done;
}
Transition;

typedef struct {
    Transition data[BUFFER_SIZE];
    int count;
    int write_pos;
}
ReplayBuffer;

static void rb_init(ReplayBuffer *rb)
{
    rb->count = 0;
    rb->write_pos = 0;
}

static void rb_add(ReplayBuffer *rb, Transition t)
{
    rb->data[rb->write_pos] = t;
    rb->write_pos = (rb->write_pos + 1) % BUFFER_SIZE;
    if (rb->count < BUFFER_SIZE) rb->count++;
}

static void rb_sample(const ReplayBuffer *rb, 
                      Transition *batch, int n)
{
    int i;
    for (i = 0; i < n; i++)
        batch[i] = rb->data[rand() % rb->count];
}

/* The Chapter 28 environment, so the buffer
   holds transitions that could really happen */
#define N_STATES 5
#define GOAL 4

static int env_step(int s, int a, float *r, 
                    int *done)
{
    int sn = (a == 1) ? s + 1 : s - 1;
    if (sn < 0) sn = 0;      /* wall on the left */
    if (sn > GOAL) sn = GOAL;
    *done = (sn == GOAL);
    *r = *done ? 10.0f : -1.0f;
    return sn;
}

int main(void)
{
    ReplayBuffer rb;
    rb_init(&rb);

    srand(42);

    /* Run episodes with a random policy and store
       every transition the environment produced */
    int i, ep, s = 0;
    for (ep = 0; ep < 6; ep++) {
        int step;
        s = 0;
        for (step = 0; step < 10; step++) {
            /* Lean right so some episodes reach
               the goal and land a done in the buffer */
            int a = (rand() % 10 < 7) ? 1 : 0;
            float r;
            int done;
            int sn = env_step(s, a, &r, &done);
            Transition t;
            t.state = s;
            t.action = a;
            t.reward = r;
            t.next_state = sn;
            t.done = done;
            rb_add(&rb, t);
            s = sn;
            if (done) break;
        }
    }

    printf("Replay buffer: %d transitions stored\n\n",
           rb.count);

    Transition batch[BATCH_SIZE];
    rb_sample(&rb, batch, BATCH_SIZE);

    printf("Random batch of %d transitions\n\n",
           BATCH_SIZE);
    printf("  state  action  reward  next  done\n");
    printf("  -----  ------  ------  ----  ----\n");
    for (i = 0; i < BATCH_SIZE; i++)
        printf("  %3d    %-6s  %+5.0f    %3d   %d\n",
               batch[i].state, 
               batch[i].action ? "right" : "left",
               batch[i].reward, batch[i].next_state, 
               batch[i].done);

    /* Report what the whole buffer holds, so the
       point does not depend on which rows were drawn */
    int terminal = 0;
    float best = -1e9f;
    for (i = 0; i < rb.count; i++) {
        if (rb.data[i].done) terminal++;
        if (rb.data[i].reward > best)
            best = rb.data[i].reward;
    }
    printf("\n  Buffer holds %d "
           "transitions, of which\n",
           rb.count);
    printf("  %d reached the goal. "
           "Best reward stored\n",
           terminal);
    printf("  is %+.0f. Any batch may "
           "or may not draw\n",
           best);
    printf("  one of those %d, and over many batches\n",
           terminal);
    printf("  each of them is reused many times.\n");

    printf("\n  Every row obeys the environment. "
           "Moving\n");
    printf("  right adds one, moving left subtracts\n");
    printf("  one and stops at the "
           "wall, and done is\n");
    printf("  set only when next is the goal.\n\n");

    printf("  The rows arrive in "
           "shuffled order, so a\n");
    printf("  batch carries no trace "
           "of the episodes\n");
    printf("  it came from. That is "
           "the whole point,\n");
    printf("  since consecutive steps "
           "are correlated\n");
    printf("  and a network trained "
           "on them wobbles.\n");

    return 0;
}
