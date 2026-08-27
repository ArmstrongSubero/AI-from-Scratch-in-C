# AI from Scratch in C

This repository contains the source code for *AI from Scratch in C: From Perceptrons to Large Language Models*, and is the
official code repository for the book.


<img width="400" height="494" alt="ai_in_c_cover" src="https://github.com/user-attachments/assets/c9b0baa3-4bc0-4baa-a2c7-bcbdd525b26a" />



In *AI from Scratch in C* you build every part of a neural network yourself, in
C, starting from a single neuron and ending with a working GPT you train on your
own machine. There is no framework, no automatic differentiation, and no library
of any kind. Every derivative in the book is worked out by hand and then written
as C you can step through in a debugger.

The approach mirrors how the real systems are built. The arithmetic in a
production transformer is the arithmetic in these programs, run at a different
scale on different hardware. Once you have written a backward pass yourself, the
frameworks stop being magic.

* [Read the book free online](https://rvembedded.com/books/ai-from-scratch-in-c/)
* [Book page](https://rvembedded.com/products/ai-from-scratch-in-c/)
* [Get the complete PDF edition](https://payhip.com/b/tCp42) (Use code LAUNCHMONTH to get the book for $29 until September 30th!)

<br>

To download a copy of this repository, click the Download ZIP button or run the
following in your terminal.

```bash
git clone --depth 1 https://github.com/ArmstrongSubero/AI-from-Scratch-in-C.git
```

<br>

## Table of Contents

Chapters 1 to 37 are free to read online. Each one is self contained and adds to
the program before it, so reading in order costs nothing and skipping around
will leave gaps.

Programs are numbered across the whole book rather than per chapter, so the
folder reads top to bottom in the same order as the text. `000_Weighted_Sum.c`
is the first program in Chapter 1 and `178_Capstone.c` is the last program in
Chapter 37.

| Chapter | Title | Programs |
| --- | --- | --- |
| 1 | [The Perceptron](https://rvembedded.com/books/ai-from-scratch-in-c/ch01.html) | 000 to 006 |
| 2 | [The Multi-Layer Perceptron](https://rvembedded.com/books/ai-from-scratch-in-c/ch02.html) | 007 to 012 |
| 3 | [Activation Functions](https://rvembedded.com/books/ai-from-scratch-in-c/ch03.html) | 013 to 018 |
| 4 | [Loss Functions](https://rvembedded.com/books/ai-from-scratch-in-c/ch04.html) | 019 to 025 |
| 5 | [Gradient Descent and Optimizers](https://rvembedded.com/books/ai-from-scratch-in-c/ch05.html) | 026 to 032 |
| 6 | [Overfitting and Regularization](https://rvembedded.com/books/ai-from-scratch-in-c/ch06.html) | 033 to 038 |
| 7 | [Hidden Representations](https://rvembedded.com/books/ai-from-scratch-in-c/ch07.html) | 039 to 043 |
| 8 | [Convolution and Filters](https://rvembedded.com/books/ai-from-scratch-in-c/ch08.html) | 044 to 049 |
| 9 | [CNN Architecture](https://rvembedded.com/books/ai-from-scratch-in-c/ch09.html) | 050 to 055 |
| 10 | [Embeddings](https://rvembedded.com/books/ai-from-scratch-in-c/ch10.html) | 056 to 060 |
| 11 | [Tokenization](https://rvembedded.com/books/ai-from-scratch-in-c/ch11.html) | 061 to 066 |
| 12 | [Sequence Problems](https://rvembedded.com/books/ai-from-scratch-in-c/ch12.html) | 067 to 071 |
| 13 | [Recurrent Neural Networks](https://rvembedded.com/books/ai-from-scratch-in-c/ch13.html) | 072 to 076 |
| 14 | [The Vanishing Gradient Problem](https://rvembedded.com/books/ai-from-scratch-in-c/ch14.html) | 077 to 080 |
| 15 | [LSTM](https://rvembedded.com/books/ai-from-scratch-in-c/ch15.html) | 081 to 086 |
| 16 | [GRU](https://rvembedded.com/books/ai-from-scratch-in-c/ch16.html) | 087 to 091 |
| 17 | [Bidirectional RNNs](https://rvembedded.com/books/ai-from-scratch-in-c/ch17.html) | 092 to 095 |
| 18 | [Sequence-to-Sequence](https://rvembedded.com/books/ai-from-scratch-in-c/ch18.html) | 096 to 100 |
| 19 | [Dot-Product Attention](https://rvembedded.com/books/ai-from-scratch-in-c/ch19.html) | 101 to 105 |
| 20 | [Query/Key/Value Attention](https://rvembedded.com/books/ai-from-scratch-in-c/ch20.html) | 106 to 109 |
| 21 | [Multi-Head Attention](https://rvembedded.com/books/ai-from-scratch-in-c/ch21.html) | 110 to 114 |
| 22 | [Positional Encoding](https://rvembedded.com/books/ai-from-scratch-in-c/ch22.html) | 115 to 119 |
| 23 | [Normalization Deep Dive](https://rvembedded.com/books/ai-from-scratch-in-c/ch23.html) | 120 to 125 |
| 24 | [The Transformer Block](https://rvembedded.com/books/ai-from-scratch-in-c/ch24.html) | 126 to 130 |
| 25 | [The Full Transformer](https://rvembedded.com/books/ai-from-scratch-in-c/ch25.html) | 131 to 134 |
| 26 | [Decoder-Only Transformer](https://rvembedded.com/books/ai-from-scratch-in-c/ch26.html) | 135 to 138 |
| 27 | [KV Cache](https://rvembedded.com/books/ai-from-scratch-in-c/ch27.html) | 139 to 143 |
| 28 | [Reinforcement Learning](https://rvembedded.com/books/ai-from-scratch-in-c/ch28.html) | 144 to 148 |
| 29 | [Q-Learning](https://rvembedded.com/books/ai-from-scratch-in-c/ch29.html) | 149 to 151 |
| 30 | [Deep Q-Networks](https://rvembedded.com/books/ai-from-scratch-in-c/ch30.html) | 152 to 154 |
| 31 | [Pretraining and Fine-Tuning](https://rvembedded.com/books/ai-from-scratch-in-c/ch31.html) | 155 to 156 |
| 32 | [Scaling Laws](https://rvembedded.com/books/ai-from-scratch-in-c/ch32.html) | 157 to 160 |
| 33 | [RLHF and Alignment](https://rvembedded.com/books/ai-from-scratch-in-c/ch33.html) | 161 to 163 |
| 34 | [LoRA](https://rvembedded.com/books/ai-from-scratch-in-c/ch34.html) | 164 to 168 |
| 35 | [Quantization](https://rvembedded.com/books/ai-from-scratch-in-c/ch35.html) | 169 to 172 |
| 36 | [Mixture of Experts](https://rvembedded.com/books/ai-from-scratch-in-c/ch36.html) | 173 to 177 |
| 37 | [A Complete GPT from Scratch](https://rvembedded.com/books/ai-from-scratch-in-c/ch37.html) | 178 |

[Appendix A](https://rvembedded.com/books/ai-from-scratch-in-c/appendix_a.html) is a derivative reference covering every
backward pass in the book. [Appendix B](https://rvembedded.com/books/ai-from-scratch-in-c/appendix_b.html) covers building
and running the code. Neither has programs of its own.

<br>

## Prerequisites

The one real prerequisite is C. You should be comfortable with pointers,
structs, arrays and `malloc`. If you have written a linked list and freed it
correctly, you are ready.

Calculus is not assumed. Every derivative the book needs is explained
immediately before it gets used, at the point where you can see what it is for.
The same is true of the linear algebra, which never goes past multiplying and
adding numbers in a loop.

Prior machine learning experience is not assumed either. If you already know the
theory, the code will be the new part.

<br>

## Requirements

Any C compiler. GCC, Clang and MSVC all work.

Every program is a single translation unit with no dependencies beyond the
standard library, so there is no build system, no package manager and no
environment to configure. Compile one file and run it.

```bash
gcc 000_Weighted_Sum.c -o weighted_sum -lm
./weighted_sum
```

The `-lm` flag links the math library and is needed on Linux and macOS for any
program calling `expf`, `tanhf` or `sqrtf`. On Windows with MSVC it is not
needed.

Everything runs on an ordinary laptop in a reasonable time. Nothing here needs a
GPU, and nothing here will use one. The capstone in Chapter 37 is the longest
running program in the book and it still finishes on CPU in just over 10 seconds. 

Appendix B covers building on each platform in more detail, including the
compiler flags worth turning on while you are working through the exercises.

<br>

## Reproducing the figures

Every figure in the book showing program output is a capture of a real run
rather than a reconstruction. Where a result depends on the platform random
number generator, the figures come from a Windows build, so your own numbers
will differ while the behaviour they demonstrate stays the same.

Nothing in the book is tuned for speed. The programs are written to be read, and
several of them do the same work two or three ways so you can watch the
difference. If a loop looks like it could be faster, it could, and making it
faster is often the exercise.

<br>

## Exercises

Every chapter ends with exercises. Most of them ask you to change one thing in
the program you just built and predict what happens before you run it, which is
the part that does the work.

Solutions are deliberately not included. The programs are short enough that an
answer would remove the whole benefit.

<br>

## Questions, feedback and corrections

Issues and Discussions on this repository are the best places for questions
about the code.

Corrections to the book itself are welcome and get credited. Send them to
armstrongsubero@gmail.com. Everything here was compiled and run before it was
printed, but a book this size will still have errors in it, and finding one is a
favour.

Since this repository holds the code for a published book, changes that alter
what a program does are hard to accept, because the listing in the text and the
file here have to agree. Fixes for genuine bugs and portability problems are
another matter and are very welcome.

<br>

## Citation

If you find the book or this code useful, please consider citing it.

Chicago style.

Subero, Armstrong. *AI from Scratch in C: From Perceptrons to Large Language
Models*. 2026.

BibTeX.

```bibtex
@book{ai-from-scratch-in-c,
  author    = {Armstrong Subero},
  title     = {AI from Scratch in C: From Perceptrons to Large Language Models},
  year      = {2026},
  url       = {https://rvembedded.com/books/ai-from-scratch-in-c/}
}
```

<br>

## License

The code in this repository is provided under the license in [LICENSE](LICENSE).
You may use it in your own projects, including commercial ones, under those
terms.

The text of the book is copyright 2026 Armstrong Subero and is not covered by
that license.
