nicslu:
	icc -O3 nicslu.c -o nicslu -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/
flu:
	icc main.c -O3 -o flu -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -lflu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -fopenmp

