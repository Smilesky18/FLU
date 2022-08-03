1. Compile command of NicSLU: rm nicslu; make nicslu 
   Compile command of FLU: rm flu; make flu

2. Example of Run command: ./nicslu ASIC_100k.mtx
   Example of Run command: ./flu ASIC_100k.mtx

3. FLU 函数中的各个参数含义 (ai, ap, ax, row_ptr_L, offset_L, L, row_ptr_U, offset_U, U, n, nnz, lnz, unz, perm_c, perm_r, thread, loop)
    ai, ap, ax: 矩阵 A 的 CSC 压缩存储 (A = L*U)；
    row_ptr_L, offset_L, L: 矩阵 L 的 CSC 压缩存储，数组 L 存储的就是要计算的值；
    row_ptr_U, offset_U, U: 矩阵 U 的 CSC 压缩存储，数组 U 存储的就是要计算的值；
    n, nnz, lnz, unz: 矩阵 A 的维度，矩阵 A 的非零元个数，矩阵 L 的非零元个数，矩阵 U 的非零元个数；
    perm_c: j = perm_c[i], 排序后矩阵的第 i 列是原矩阵的第 j 列;
    perm_r: j = perm_r[i], 排序后矩阵的第 j 行是原矩阵的第 i 行;
    thread: 当该值为 0 时，线程数目为 32 (和 NicSLU 中的机制一样);
    loop: 函数运行的总次数。
