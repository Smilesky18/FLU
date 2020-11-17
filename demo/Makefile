nicslu:
	icc -O3 demo.c -o nicslu -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/
flu:
	icc -g main_plan_3.c computing_plan_2_next.c -O3 -o flu -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -fopenmp #-I. -L. -lmetis
flu_double_col:
	icc -g main_plan_3_double_column.c computing_plan_2_next_double_column.c -O3 -o flu_double_col -I ../include/ -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/ -fopenmp
pardiso:
	icc -m64  -w -DMKL_ILP64 -I ../include pardiso_unsym_csc_c.c -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -L ../linux/lib_centos6_x64_gcc482_fma/int32/ -liomp5 -lpthread -lm -ldl -lnicslu -o pardiso -Wl,-rpath=../linux/lib_centos6_x64_gcc482_fma/int32/
clean:
	rm -rf flu
