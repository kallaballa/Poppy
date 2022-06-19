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
CXXFLAGS += `pkg-config --cflags opencv4 libpng` -I../third/gif-h/ -march=native
LIBS += -lboost_program_options `pkg-config --libs opencv4 libpng sdl SDL_image`
else
CXX     := em++
CXXFLAGS += -D_WASM -D_NO_FACE_DETECT -I../third/gif-h/ -I../third/opencv-4.6.0/modules/core/include -I../third/build_wasm/ -I../third/opencv-4.6.0/modules/imgproc/include/ -I../third/opencv-4.6.0/modules/features2d/include/ -I../third/opencv-4.6.0/modules/flann/include/ -I../third/opencv-4.6.0/modules/videoio/include/ -I../third/opencv-4.6.0/modules/highgui/include/  -I../third/opencv-4.6.0/modules/calib3d/include/ -I../third/opencv-4.6.0/modules/video/include/ -I../third/opencv-4.6.0/modules/imgcodecs/include/ -I../third/opencv-4.6.0/include/ -I../third/opencv-4.6.0/modules/dnn/include/ -I../third/opencv-4.6.0/modules/photo/include/ -I../third/opencv-4.6.0/modules/objdetect/include -I../third/opencv_contrib-4.x/modules/face/include -I../third/opencv-4.6.0/modules/ml/include/ -I../third/opencv-4.6.0/modules/stitching/include
LDFLAGS += -L../third/build_wasm/lib/ -L../third/build_wasm/3rdparty/lib/
EMCXXFLAGS += -flto -s USE_PTHREADS=1 -s NO_DISABLE_EXCEPTION_CATCHING
EMLDFLAGS += -s WASM=1 -D_WASM -s USE_PTHREADS=1 -s ALLOW_MEMORY_GROWTH=1 -s WASM_BIGINT -s STB_IMAGE=1 -s "EXPORTED_FUNCTIONS=['_load_images', '_main' ]" -s EXPORTED_RUNTIME_METHODS='["ccall" ]' -s LLD_REPORT_UNDEFINED=1 -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency -s PROXY_TO_PTHREAD=1 -s ALLOW_MEMORY_GROWTH=1 -s NO_DISABLE_EXCEPTION_CATCHING -s EXCEPTION_DEBUG=1
#--preload-file assets
LIBS += -lzlib -lopencv_calib3d -lopencv_core -lopencv_dnn -lopencv_features2d -lopencv_flann -lopencv_imgproc -lopencv_objdetect -lopencv_photo -lopencv_video -lopencv_objdetect -lopencv_face
#EMCXXFLAGS += -msimd128 -msse2
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
#asan: CXXFLAGS += -fsanitize=address
asan: LDFLAGS += -s STACK_OVERFLOW_CHECK=2 -s ASSERTIONS=2
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
export WASM

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
