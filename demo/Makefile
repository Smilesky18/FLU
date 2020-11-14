nicslu:
	icc -O3 demo.c -o nicslu -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/
flu:
	icc main_plan_3.c computing_plan_2_next.c -O3 -o flu -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -fopenmp

clean:
	rm -rf flu
