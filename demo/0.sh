path=$1
files=$(ls $path)

for filename in $files
do
	#perf stat -e LLC-stores -o LLC-stores-nicslu --append ./demo $path/$filename 
	./flu $path/$filename 32 100 8 4
done
