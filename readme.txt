cd /home/username/xiebw/github/FLU/demo
run:
	make flu
	make nicslu
	make pardiso

Preprocess for compiling pardiso:
	sudo cp -r [intel]/compilers_and_libraries_2019.5.281/linux/mkl/include/* /usr/include/
	sudo cp -r [intel]/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64/* /usr/lib/
	sudo cp -r [intel]/compilers_and_libraries_2019.5.281/linux/mkl/../compiler/lib/intel64/* /usr/lib/
