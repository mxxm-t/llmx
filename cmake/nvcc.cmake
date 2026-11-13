# Runs nvcc on one kernel file and keeps ptxas's report of each target beside the fatbin, which the SASS witnesses read (tests/cuda_sass.py).
# cmake -DNVCC=nvcc -DSRC=file.gpu -DOUT=file.fatbin -DREPORT=file.ptxas -DINCLUDE=dir -DARCHS="80;86" -DPTX=80 -P nvcc.cmake
set(gencode)
foreach(a IN LISTS ARCHS)
    list(APPEND gencode -gencode arch=compute_${a},code=sm_${a})
endforeach()
list(APPEND gencode -gencode arch=compute_${PTX},code=compute_${PTX})
execute_process(
    COMMAND ${NVCC} -x cu -std=c++17 -O3 --expt-relaxed-constexpr --fmad=false -prec-div=true -prec-sqrt=true -ftz=false --threads 0
            ${gencode} -Xptxas -v -I${INCLUDE} -fatbin ${SRC} -o ${OUT}
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
file(WRITE ${REPORT} "${out}${err}")
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "nvcc failed on ${SRC}:\n${err}")
endif()
