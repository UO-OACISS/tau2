HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
ROCM_DIR = $(shell $(HIP_PATH)/bin/hipconfig --rocmpath)
HIP_PATH ?= $(ROCM_DIR)/hip
BUILD_DIR ?= .

HIPCC = ${HIP_PATH}/bin/hipcc
HIPCC = hipcc
CPPFLAGS = -O3
LDFLAGS = -lm -lpthread

ifeq (${HIP_PLATFORM}, nvcc)
    CPPFLAGS += -arch=compute_20
endif

GPUBURN_SRC = $(wildcard *.cpp)
GPUBURN_OBJ = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(GPUBURN_SRC)))
GPUBURN_BIN = ${BUILD_DIR}/gpuburn-hip

.PHONY: all clean run itburn

all: ${GPUBURN_BIN}

${GPUBURN_BIN}: ${GPUBURN_OBJ}
	${HIPCC} ${LDFLAGS} -o ${GPUBURN_BIN} ${GPUBURN_OBJ}

${BUILD_DIR}/%.o: %.cpp Makefile
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	$(HIPCC) ${HIP_INCLUDE} ${CPPFLAGS} -c -o $@ $<
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

run: itburn
itburn:
	HCC_LAZYINIT=ON ${GPUBURN_BIN}

clean:
	rm -rf ${BUILD_DIR}/*.o ${GPUBURN_BIN} profile*
