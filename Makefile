CXX      := g++
CXXFLAGS := -std=c++17 -pthread -pedantic -Wall -fno-strict-aliasing
LDFLAGS  := -L/opt/local/lib 
LIBS     := -lm -lpthread
.PHONY: all release debian-release info debug clean debian-clean distclean asan shrink
ESTDIR := /
PREFIX := /usr/local
MACHINE := $(shell uname -m)
UNAME_S := $(shell uname -s)
LIBDIR := lib

UNAME_P := $(shell uname -p)

ifeq ($(UNAME_S), Darwin)
 LDFLAGS += -L/opt/X11/lib/
endif

ifeq ($(UNAME_P),x86_64)
LIBDIR = lib64
endif

ifndef TIMETRACK
CXXFLAGS += -D_NO_TIMETRACK
endif

ifdef NOTHREADS
CXXFLAGS += -D_NO_THREADS
endif

ifdef AUTOVECTOR
CXXFLAGS += -D_AUTOVECTOR
endif

ifndef WASM
CXXFLAGS += -march=native `pkg-config --cflags opencv4 libpng`
LIBS += -lboost_program_options `pkg-config --libs opencv4 libpng`
else
CXX     := em++
CXXFLAGS += -D_WASM -s USE_PTHREADS=1 -I../third/opencv/
LDFLAGS += -D_WASM -s USE_PTHREADS=1 -s PROXY_TO_PTHREAD -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency
EMCXXFLAGS += -flto -s DISABLE_EXCEPTION_CATCHING=1
EMLDFLAGS += --js-library ../third/opencv.js -s INITIAL_MEMORY=419430400 -s ASYNCIFY -s TOTAL_STACK=52428800 -s WASM_BIGINT -s MALLOC=emmalloc

ifdef AUTOVECTOR
EMCXXFLAGS += -msimd128
endif

CXXFLAGS += $(EMCXXFLAGS) -c
LDFLAGS += $(EMLDFLAGS)
endif

ifdef X86
CXXFLAGS += -m32
LDFLAGS += -L/usr/lib -m32 -static-libgcc -m32 -Wl,-Bstatic
endif

ifdef STATIC
LDFLAGS += -static-libgcc -Wl,-Bstatic
endif

all: release

ifneq ($(UNAME_S), Darwin)
release: LDFLAGS += -s
endif
ifdef WASM
release: LDFLAGS += -s STACK_OVERFLOW_CHECK=2 -s ASSERTIONS=0 -s SAFE_HEAP=0
endif
release: CXXFLAGS += -g0 -O3 -c
release: dirs

shrink: CXXFLAGS += -Os -w
shrink: LDFLAGS += -s
shrink: dirs

info: CXXFLAGS += -g3 -O0
info: LDFLAGS += -Wl,--export-dynamic -rdynamic
info: dirs

ifndef WASM
debug: CXXFLAGS += -rdynamic
debug: LDFLAGS += -rdynamic
endif
debug: CXXFLAGS += -g3 -O0
debug: LDFLAGS += -Wl,--export-dynamic
debug: dirs

profile: CXXFLAGS += -g3 -O3 
profile: LDFLAGS += -Wl,--export-dynamic
ifdef WASM
profile: LDFLAGS += --profiling
profile: CXXFLAGS += --profiling
endif
ifndef WASM
profile: CXXFLAGS += -rdynamic
endif
profile: dirs

ifdef WASM
hardcore: CXXFLAGS += -DNDEBUG -g0 -O3 -ffp-contract=fast -freciprocal-math -fno-signed-zeros --closure 1 
hardcore: LDFLAGS += -s STACK_OVERFLOW_CHECK=0 -s ASSERTIONS=0 -s SAFE_HEAP=0 --closure 1 -menable-unsafe-fp-math
else
hardcore: CXXFLAGS += -DNDEBUG -g0 -Ofast
endif
#ifeq ($(UNAME_S), Darwin)
hardcore: LDFLAGS += -s
#endif
hardcore: dirs

ifdef WASM
#asan: CXXFLAGS += -fsanitize=undefined
asan: LDFLAGS += -s STACK_OVERFLOW_CHECK=2 -s ASSERTIONS=2 #-fsanitize=undefined
else
asan: CXXFLAGS += -rdynamic -fsanitize=address
asan: LDFLAGS += -rdynamic -fsanitize=address
endif
asan: CXXFLAGS += -g3 -O0 -fno-omit-frame-pointer
asan: LDFLAGS += -Wl,--export-dynamic 
ifndef WASM
asan: LIBS+= -lbfd -ldw
endif
asan: dirs

clean: dirs

export LDFLAGS
export CXXFLAGS
export LIBS

dirs:
	${MAKE} -C src/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
#	${MAKE} -C exp/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}

debian-release:
	${MAKE} -C src/ -${MAKEFLAGS} CXX=${CXX} release
#	${MAKE} -C exp/ -${MAKEFLAGS} CXX=${CXX} release
	
debian-clean:
	${MAKE} -C src/ -${MAKEFLAGS} CXX=${CXX} clean
#	${MAKE} -C exp/ -${MAKEFLAGS} CXX=${CXX} clean
	
install: ${TARGET}
	true
	
distclean:
	true
