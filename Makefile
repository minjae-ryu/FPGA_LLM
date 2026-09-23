# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c -lm
	$(CC) -O3 -o runq runq.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.c
	$(CC) -g -o run run.c -lm
	$(CC) -g -o runq runq.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.c
	$(CC) -Ofast -o run run.c -lm
	$(CC) -Ofast -o runq runq.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(CC) -Ofast -fopenmp -march=native run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -march=native runq.c  -lm  -o runq

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run.exe -I. run.c win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. runq.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(CC) -Ofast -std=gnu11 -o run run.c -lm
	$(CC) -Ofast -std=gnu11 -o runq runq.c -lm

.PHONY: runompgnu
runompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -std=gnu11 runq.c  -lm  -o runq

# run all tests
.PHONY: test
test:
	pytest

# run only tests for run.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f run
	rm -f runq

# SmolLM2 accuracy build. The legacy targets above intentionally stay unchanged.
SM_CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -fopenmp -march=native
SM_CPPFLAGS = -Iinclude $(shell pkg-config --cflags openblas libpcre2-8) -DSM_BUILD_FLAGS='"$(SM_CFLAGS)"'
SM_LIBS = $(shell pkg-config --libs openblas libpcre2-8) -lm -fopenmp
SM_SOURCES = src/model.c src/session.c src/kernels.c src/cache.c src/math_ops.c src/scoring.c src/sampling.c src/cuda_stub.c
SM_COMMANDS = src/cli_common.c src/main.c src/generate.c src/tokenizer.c src/evaluate.c src/benchmark.c src/replay.c src/compare_results.c
SM_HEADERS = include/smollm.h include/sm_cuda.h src/internal.h
PYTHON ?= python3

build:
	mkdir -p build

build/libsmollm.so: $(SM_SOURCES) $(SM_HEADERS) | build
	$(CC) $(SM_CFLAGS) $(SM_CPPFLAGS) -fPIC -shared $(SM_SOURCES) -o $@ $(SM_LIBS)

.PHONY: smollm check
build/smollm: $(SM_SOURCES) $(SM_HEADERS) include/sm_cli.h include/sm_tokenizer.h $(SM_COMMANDS) | build
	$(CC) $(SM_CFLAGS) $(SM_CPPFLAGS) $(SM_SOURCES) $(SM_COMMANDS) -o $@ $(SM_LIBS)

smollm: build/libsmollm.so build/smollm

check: smollm
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) -m pytest -q tests

# Optional CUDA foundation. C CLI dispatch is a later milestone; this builds
# the C ABI library, not a CLI claiming to execute CUDA forward.
NVCC ?= nvcc
CUDA_GENCODE ?= -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86
CUDA_FLAGS ?= -O3 -std=c++14 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false
CUDA_COMPILE = $(NVCC) $(CUDA_FLAGS) $(CUDA_GENCODE) -Iinclude -Xcompiler=-fPIC,-Wall,-Wextra,-fno-fast-math,-ffp-contract=off
CUDA_SOURCES = src/cuda/model.cu src/cuda/session.cu
CUDA_OBJECTS = $(patsubst src/cuda/%.cu,build/cuda/%.o,$(CUDA_SOURCES))
CUDA_C_OBJECTS = $(patsubst src/%.c,build/cuda/c/%.o,$(filter-out src/cuda_stub.c,$(SM_SOURCES)))

build/cuda/%.o: src/cuda/%.cu src/cuda/internal.cuh include/sm_cuda.h include/smollm.h
	mkdir -p $(@D)
	$(CUDA_COMPILE) -c $< -o $@

build/cuda/c/%.o: src/%.c $(SM_HEADERS)
	mkdir -p $(@D)
	$(CC) $(SM_CFLAGS) $(SM_CPPFLAGS) -fPIC -c $< -o $@

build/libsmollm-cuda.so: $(CUDA_OBJECTS) $(CUDA_C_OBJECTS)
	$(NVCC) -shared $^ -o $@ -lcublas $(filter-out -fopenmp,$(SM_LIBS)) -Xcompiler=-fopenmp

build/cuda/bootstrap-c.o: tests/cuda_bootstrap.c include/smollm.h
	mkdir -p $(@D)
	$(CC) $(SM_CFLAGS) -Iinclude -c $< -o $@

build/cuda/bootstrap.o: tests/cuda_bootstrap.cu src/cuda/internal.cuh include/sm_cuda.h include/smollm.h
	mkdir -p $(@D)
	$(CUDA_COMPILE) -c $< -o $@

build/cuda-bootstrap: build/cuda/bootstrap-c.o build/cuda/bootstrap.o
	$(NVCC) $^ -o $@ -lcublas

.PHONY: cuda check-cuda
cuda: build/libsmollm-cuda.so build/cuda-bootstrap
check-cuda: smollm cuda build/cuda-lifecycle
	CUDA_CACHE_PATH=$(CURDIR)/.cache/cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SM_TEST_CUDA=1 $(PYTHON) -m pytest -q tests/test_cuda.py

build/cuda-test/%.o: src/cuda/%.cu src/cuda/internal.cuh include/sm_cuda.h include/smollm.h
	mkdir -p $(@D)
	$(CUDA_COMPILE) -DSM_CUDA_TESTING -c $< -o $@

build/cuda-test/lifecycle.o: tests/cuda_lifecycle.cu src/cuda/internal.cuh include/sm_cuda.h include/smollm.h
	mkdir -p $(@D)
	$(CUDA_COMPILE) -DSM_CUDA_TESTING -c $< -o $@

build/cuda-lifecycle: build/cuda-test/lifecycle.o build/cuda-test/model.o build/cuda-test/session.o build/cuda/c/model.o
	$(NVCC) $^ -o $@ -lcublas
