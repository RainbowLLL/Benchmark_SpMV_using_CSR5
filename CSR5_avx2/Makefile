#compilers
CC=c++

#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 0

#ENVIRONMENT_PARAMETERS
#OPRIONS=-xCORE-AVX2 -opt-prefetch=3 #icc/icpc option
OPRIONS=-mavx -mavx2 -march=native #c++ option
#backup
#$(CC) -xCORE-AVX2 -opt-prefetch=3 -Wno-deprecated-writable-strings -fopenmp -O3 main.cpp -o spmv -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)

all: clean avx2 clear exec
.PHONY : all

clean:
	rm -f ./spmv
avx2:
	$(CC) $(OPRIONS) -fopenmp -g -O3 main.cpp -o spmv -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)

clear:
	clear
	clear

exec:
	./spmv ~/data/sparse_perf_matrix/csr5_eg_1/csr5_eg_1.mtx | tee result.log

