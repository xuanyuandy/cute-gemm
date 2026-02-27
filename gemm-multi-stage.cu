#include <cublas_v2.h>
#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>

#include <cute/tensor.hpp>

#include "detail/cublaslt-gemm.h"
#include "detail/data.h"

// CuTe 多阶段流水线 GEMM 实现
//
// 数据流:
//   全局内存 (gmem) -cp.async-> 共享内存 (shm) -ldmatrix-> 寄存器 (reg)
//
// 核心优化技术:
//   1. 多阶段流水线 (Multi-stage pipeline): 使用 kStage 个缓冲区隐藏内存访问延迟
//   2. 异步复制 (cp.async): 全局内存到共享内存的异步复制
//   3. ldmatrix 指令: 共享内存到寄存器的高效加载
//   4. Swizzling: 避免共享内存 bank conflict
//   5. Tile 分块: 将大矩阵分解为小 tile 进行计算
//
// 张量形状说明:
//   (CPY, CPY_M, CPY_K, kStage)
//   - CPY: 每个线程拥有的复制元素数量
//   - CPY_M/N/K: 在 M/N/K 方向上的重复次数
//   - kStage: 流水线阶段数

template <typename Config>
__global__ void /* __launch_bounds__(128, 1) */
gemm_multi_stage(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
                 int k) {
  using namespace cute;
  using X = Underscore;

  // 从配置中提取类型定义
  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  // 复制原子类型
  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  // 使用张量表示法表示设备指针 + 维度
  // A: 输入矩阵 A，形状 (M, K)，列主序布局
  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  // B: 输入矩阵 B，形状 (N, K)，列主序布局
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  // D: 输出矩阵 D，形状 (M, N)，列主序布局
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  // 将全局张量切片为当前线程块使用的小张量
  // gA: 当前线程块的矩阵 A tile，形状 (kTileM, kTileK, k)
  // - kTileM: M 维度的 tile 大小
  // - kTileK: K 维度的 tile 大小
  // - k: K 维度的 tile 数量
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _));
  // gB: 当前线程块的矩阵 B tile，形状 (kTileN, kTileK, k)
  // - kTileN: N 维度的 tile 大小
  // - kTileK: K 维度的 tile 大小
  // - k: K 维度的 tile 数量
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _));
  // gD: 当前线程块的输出矩阵 D tile，形状 (kTileM, kTileN)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix));

  // 共享内存张量
  // sA: 矩阵 A 的共享内存，形状 (kTileM, kTileK, kStage)
  // - kStage: 流水线阶段数，用于多缓冲
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{});
  // sB: 矩阵 B 的共享内存，形状 (kTileN, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{});

  // 将 TileA/TileB/TileC MMA 张量分发到线程片段
  // tiled_mma: 分块的 MMA 操作
  TiledMMA tiled_mma;
  // thr_mma: 当前线程的 MMA 切片
  auto thr_mma = tiled_mma.get_slice(idx);
  
  // tCrA: 当前线程的矩阵 A 寄存器片段，形状 (MMA, MMA_M, MMA_K)
  // - MMA: MMA 操作数量
  // - MMA_M: 每个 MMA 操作的 M 维度
  // - MMA_K: 每个 MMA 操作的 K 维度
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  
  // tCrB: 当前线程的矩阵 B 寄存器片段，形状 (MMA, MMA_N, MMA_K)
  // - MMA: MMA 操作数量
  // - MMA_N: 每个 MMA 操作的 N 维度
  // - MMA_K: 每个 MMA 操作的 K 维度
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  
  // tCrD: 当前线程的输出矩阵 D 寄存器片段（累加器），形状 (MMA, MMA_M, MMA_N)
  // - MMA: MMA 操作数量
  // - MMA_M: 每个 MMA 操作的 M 维度
  // - MMA_N: 每个 MMA 操作的 N 维度
  auto tCrD = thr_mma.partition_fragment_C(gD);

  // 将累加器清零
  clear(tCrD);

  // gmem -cp.async-> shm -ldmatrix-> reg
  // CPY means the element: 每个线程在一次复制操作中需要处理的数据元素个数
  // CPY_M: 在 M 维度方向上的重复次数
  // CPY_K: 在 K 维度方向上的重复次数
  // kStage: 多流水线阶段的数量，用于数据预取和双缓冲/多缓冲技术
  
  // 共享内存到寄存器复制 (S2R) - 矩阵 A
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  // tAsA: 从共享内存读取的源张量视图
  // 形状: (CPY, CPY_M, CPY_K, kStage)
  // - CPY: 每个线程拥有的复制元素数量
  // - CPY_M: 在 M 方向上的重复次数
  // - CPY_K: 在 K 方向上的重复次数  
  // - kStage: 流水线阶段维度
  auto tAsA = s2r_thr_copy_a.partition_S(sA);
  
  // tCrA_view: 寄存器目标张量视图，用于 ldmatrix 指令
  // 形状: (CPY, CPY_M, CPY_K)
  // - CPY: 每个线程拥有的复制元素数量
  // - CPY_M: 在 M 方向上的重复次数
  // - CPY_K: 在 K 方向上的重复次数
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

  // 共享内存到寄存器复制 (S2R) - 矩阵 B
  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  // tBsB: 从共享内存读取的源张量视图
  // 形状: (CPY, CPY_N, CPY_K, kStage)
  // - CPY: 每个线程拥有的复制元素数量
  // - CPY_N: 在 N 方向上的重复次数
  // - CPY_K: 在 K 方向上的重复次数
  // - kStage: 流水线阶段维度
  auto tBsB = s2r_thr_copy_b.partition_S(sB);
  
  // tCrB_view: 寄存器目标张量视图，用于 ldmatrix 指令
  // 形状: (CPY, CPY_N, CPY_K)
  // - CPY: 每个线程拥有的复制元素数量
  // - CPY_N: 在 N 方向上的重复次数
  // - CPY_K: 在 K 方向上的重复次数
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

  // 全局内存到共享内存复制 (G2S) - 矩阵 A
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  // tAgA_copy: 从全局内存读取的源张量视图
  // 形状: (CPY, CPY_M, CPY_K, k)
  // - CPY: 每个线程拥有的复制元素数量
  // - CPY_M: 在 M 方向上的重复次数
  // - CPY_K: 在 K 方向上的重复次数
  // - k: K 维度的 tile 数量
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
  
  // tAsA_copy: 写入共享内存的目标张量视图
  // 形状: (CPY, CPY_M, CPY_K, kStage)
  // - CPY: 每个线程拥有的复制元素数量
  // - CPY_M: 在 M 方向上的重复次数
  // - CPY_K: 在 K 方向上的重复次数
  // - kStage: 流水线阶段维度
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

  // 全局内存到共享内存复制 (G2S) - 矩阵 B
  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  // tBgB_copy: 从全局内存读取的源张量视图
  // 形状: (CPY, CPY_N, CPY_K, k)
  // - CPY: 每个线程拥有的复制元素数量
  // - CPY_N: 在 N 方向上的重复次数
  // - CPY_K: 在 K 方向上的重复次数
  // - k: K 维度的 tile 数量
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
  
  // tBsB_copy: 写入共享内存的目标张量视图
  // 形状: (CPY, CPY_N, CPY_K, kStage)
  // - CPY: 每个线程拥有的复制元素数量
  // - CPY_N: 在 N 方向上的重复次数
  // - CPY_K: 在 K 方向上的重复次数
  // - kStage: 流水线阶段维度
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

  // 调试信息打印（仅 thread 0 执行）
  if(thread0()) {
    // 打印 tiled MMA 配置
    cute::print(tiled_mma);
    // cute::print_latex(tiled_mma);

    // 打印全局内存布局
    printf("\n gA layout is \n");
    cute::print(gA);
    
    printf("\n tAgA_copy layout is \n");
    cute::print(tAgA_copy);

    printf("\n tAsA_copy layout is \n");
    cute::print(tAsA_copy);

    // 打印共享内存布局
    printf("\n sA layout is \n");
    cute::print(sA);

    printf("\n tAsA layout is \n");
    cute::print(tAsA);

    // 打印寄存器布局
    printf("\n tCrA layout is \n");
    cute::print(tCrA);

    printf("\n tCrA_view layout is \n");
    cute::print(tCrA_view);
    printf("\n");
    
    // 打印线程数量
    // tile copy thread num is same as tiled_mma
    int num_threads = cute::size(tiled_mma);
    printf("s2r_tiled_copy_a thread num is : %d\n", num_threads);

    printf("\n");
  }
  
  // 流水线索引
  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

  // 预热阶段: 提交 kStage - 1 个 tile 的异步复制
  // 目的: 填充流水线，隐藏后续迭代的内存访问延迟
  // 数据流: 全局内存 -> 共享内存 (使用 cp.async 异步复制)
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    // 异步复制矩阵 A: 全局内存 -> 共享内存
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    // 异步复制矩阵 B: 全局内存 -> 共享内存
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));

    if (thread0()) {
      // assert(istage >= 0 && istage < cute::size<3>(tAgA_copy));
      printf("##### total stage is %d, current stage is %d\n", kStage, istage);
      printf("\n tAgA_copy layout is \n");
      cute::print(tAgA_copy(_, _, _, istage));
      printf("\n");
    }

    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  __syncthreads();

  int ik = 0;
  // 初始加载: 共享内存 -> 寄存器
  // 加载第一个 K tile 的数据到寄存器
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // 主循环: 遍历 K 维度的所有 tile
  // 每次迭代: i. 预取下一个 tile 到寄存器, ii. 执行矩阵乘累加
  int ntile = k / kTileK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      // 在最后一个 K 维度迭代时，等待异步复制完成并同步
      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        // 更新读取的共享内存阶段索引
        ismem_read = (ismem_read + 1) % kStage;
      }

      // 预取: 共享内存 -> 寄存器
      // 加载下一个 K 维度的数据到寄存器 (流水线技术)
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      // 在第一个 K 维度迭代时，启动下一个 tile 的异步复制
      if (ik == 0) {
        if (itile_to_read < ntile) {
          // 异步复制: 全局内存 -> 共享内存
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          // 更新写入的共享内存阶段索引
          ismem_write = (ismem_write + 1) % kStage;
        }

        // 设置异步复制屏障
        cp_async_fence();
      }

      // 执行矩阵乘累加: D += A * B
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }  // for ik
  }    // itile

  // Epilogue: 使用共享内存作为暂存区，使用大宽指令
  // 数据流: Dreg -> shm -> reg -> global
  // 这样做的原因是累加器类型和输出数据类型可能不同，需要通过共享内存进行转换
  
  // sC: 复用矩阵 A 的共享内存空间作为输出矩阵 C 的暂存区
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  // 寄存器到共享内存复制 (R2S) - 矩阵 C
  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  // tCrC_r2s: 寄存器源张量视图（累加器）
  // 形状: (CPY, CPY_M, CPY_N)
  // - CPY: 每个线程拥有的复制元素数量
  // - CPY_M: 在 M 方向上的重复次数
  // - CPY_N: 在 N 方向上的重复次数
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
  
  // tCsC_r2s: 共享内存目标张量视图
  // 形状: (CPY, _1, _1, pipe)
  // - CPY: 每个线程拥有的复制元素数量
  // - _1: 单一维度
  // - pipe: 流水线维度
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

  // 共享内存到全局内存复制 (S2G) - 矩阵 C
  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  // tCsC_s2g: 共享内存源张量视图
  // 形状: (CPY, _1, _1, pipe)
  // - CPY: 每个线程拥有的复制元素数量
  // - _1: 单一维度
  // - pipe: 流水线维度
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
  
  // tCgC_s2g: 全局内存目标张量视图
  // 形状: (CPY, CPY_M, CPY_N)
  // - CPY: 每个线程拥有的复制元素数量
  // - CPY_M: 在 M 方向上的重复次数
  // - CPY_N: 在 N 方向上的重复次数
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);

  // 将后三个维度合并为一个维度，便于分批处理
  // tCgC_s2gx: 合并后的全局内存目标张量视图
  // 形状: (CPY_, CPY_MN)
  // - CPY_: 合并后的复制维度
  // - CPY_MN: 合并后的 M*N 维度
  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
  
  // tCrC_r2sx: 合并后的寄存器源张量视图
  // 形状: (CPY_, CPY_MN)
  // - CPY_: 合并后的复制维度
  // - CPY_MN: 合并后的 M*N 维度
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

  // step: 每次处理的流水线步数
  int step = size<3>(tCsC_r2s);
  
  // 分批处理输出数据，避免共享内存溢出
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    // 第一步: 寄存器 -> 共享内存
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // 创建临时张量来处理累加器和输出数据类型的差异
      // 例如: 累加器可能是 float32，而输出可能是 float16
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t);

      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    // 第二步: 共享内存 -> 全局内存
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }

    __syncthreads();
  }
}

namespace config {

using namespace cute;

template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
          int kStage_ = 5, int kSmemLayoutCBatch_ = 2,
          typename ComputeType = T_>
struct GemmConfig {
  using T = T_;

  // Tile 配置
  // kTileM: M 维度的 tile 大小
  static constexpr int kTileM = kTileM_;
  // kTileN: N 维度的 tile 大小
  static constexpr int kTileN = kTileN_;
  // kTileK: K 维度的 tile 大小
  static constexpr int kTileK = kTileK_;
  // kStage: 流水线阶段数，用于多缓冲
  static constexpr int kStage = kStage_;
  // kSmemLayoutCBatch: 共享内存布局的批次数
  static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

  // 共享内存加载的 Swizzle 参数，用于避免 bank conflict
  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  // SmemLayoutAtom: 共享内存的原子布局，使用 Swizzle 优化
  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                  make_stride(Int<kTileK>{}, Int<1>{}))));
  
  // 使用 Swizzle 布局扩展到完整的 tile 形状
  // SmemLayoutA: 矩阵 A 的共享内存布局，形状 (kTileM, kTileK, kStage)
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  // SmemLayoutB: 矩阵 B 的共享内存布局，形状 (kTileN, kTileK, kStage)
  using SmemLayoutB = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

  // MMA 操作类型: 16x8x16 的 FP16 矩阵乘累加，转置-非转置布局
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;

  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  // MMA 执行单元的重复次数
  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  // 每个 MMA tile 的 M 维度大小
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  // 每个 MMA tile 的 N 维度大小
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  // 每个 MMA tile 的 K 维度大小
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  // MMA_EU_RepeatT: 影响 MMA 执行单元的寄存器扩展
  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  // MMA_P_T: MMA tile 的形状
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

  // MMA: 分块的 MMA 操作
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  // 全局内存到共享内存的复制操作
  // g2s_copy_op: 使用 cp.async 指令，缓存到全局内存
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  // G2SCopyA: 矩阵 A 的全局到共享复制配置
  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  // G2SCopyB: 矩阵 B 的全局到共享复制配置（与 A 相同）
  using G2SCopyB = G2SCopyA;

  // 共享内存到寄存器的复制操作
  // s2r_copy_op: 使用 ldmatrix 指令加载 4 个 32 位值
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

  // S2RCopyAtomA: 矩阵 A 的共享到寄存器复制原子
  using S2RCopyAtomA = s2r_copy_atom;
  // S2RCopyAtomB: 矩阵 B 的共享到寄存器复制原子
  using S2RCopyAtomB = s2r_copy_atom;

  // Epilogue: 通过共享内存将寄存器写入全局内存
  // SmemLayoutAtomC: 矩阵 C 的共享内存原子布局，使用 Swizzle 优化
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                      make_stride(Int<kMmaPN>{}, Int<1>{}))));
  
  // SmemLayoutC: 矩阵 C 的共享内存布局，形状 (kMmaPM, kMmaPN, kSmemLayoutCBatch)
  // - kSmemLayoutCBatch: 用于分批处理输出数据，避免共享内存溢出
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  // 静态断言: 确保 C 的共享内存大小不超过 A 的一个流水线阶段的大小
  // 这样可以复用 A 的共享内存空间
  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                    size(SmemLayoutC{}),
                "C shared memory request is large than A's one pipe");

  // R2SCopyAtomC: 寄存器到共享内存的复制原子（通用复制）
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

  // S2GCopyAtomC: 共享内存到全局内存的复制原子（使用 128宽位通用复制）
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  // S2GCopyC: 矩阵 C 的共享到全局复制配置
  using S2GCopyC =
      decltype(make_tiled_copy(S2GCopyAtomC{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  // kThreadNum: 每个 thread block 的线程数，等于 MMA 操作的线程数
  static constexpr int kThreadNum = size(MMA{});
  // shm_size_AB: 矩阵 A 和 B 的共享内存总大小
  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  // shm_size_C: 矩阵 C 的共享内存大小
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});

  // kShmSize: 所需的共享内存总大小（取 AB 和 C 的最大值）
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

}  // namespace config

int main(int argc, char *argv[]) {
  // 使用 FP16 作为数据类型
  using T = cute::half_t;
  using namespace cute;
  using X = Underscore;

  srand(10086);

  // 创建 cuBLAS 句柄
  cublasHandle_t handle;
  cublasCreate(&handle);
  int cublas_version;
  cublasGetVersion_v2(handle, &cublas_version);
  printf("cuBLAS version: %d\n", cublas_version);

  // 默认矩阵大小
  int M = 128;
  int N = 128;
  int K = 32;

  int enable_cpu = 0;
  int enable_cublaslt = 1;
  int nt = 1;

  using ComputeType = T;

  // 设备内存指针
  T *Aptr;
  T *Bptr;
  T *Dptr;
  T *Dptr_cublas;
  T *Dptr_cublaslt;

  // 主机内存指针
  T *Aptr_host;
  T *Bptr_host;
  T *Dptr_host;
  T *Dptr_host_cpu;
  T *Dptr_host_blas;
  T *Dptr_host_cublaslt;

  // 分配主机内存
  Aptr_host = (T *)malloc(sizeof(T) * M * K);
  Bptr_host = (T *)malloc(sizeof(T) * N * K);
  Dptr_host = (T *)malloc(sizeof(T) * M * N);

  Dptr_host_cpu = (T *)malloc(sizeof(T) * M * N);
  Dptr_host_blas = (T *)malloc(sizeof(T) * M * N);
  Dptr_host_cublaslt = (T *)malloc(sizeof(T) * M * N);

  // 分配设备内存
  cudaMalloc(&Aptr, sizeof(T) * M * K);
  cudaMalloc(&Bptr, sizeof(T) * N * K);
  cudaMalloc(&Dptr, sizeof(T) * M * N);
  cudaMalloc(&Dptr_cublas, sizeof(T) * M * N);
  cudaMalloc(&Dptr_cublaslt, sizeof(T) * M * N);

  // 创建主机张量用于初始化
  auto tA = make_tensor(Aptr_host, make_shape(M, K), make_stride(K, 1));
  auto tB = make_tensor(Bptr_host, make_shape(N, K), make_stride(K, 1));
  auto tD = make_tensor(Dptr_host, make_shape(M, N), make_stride(N, 1));

  // 随机初始化输入矩阵
  cpu_rand_data(&tA);
  cpu_rand_data(&tB);

  // 清零输出矩阵
  clear(tD);

  // 将数据从主机复制到设备
  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * N * K, cudaMemcpyHostToDevice);
  cudaMemcpy(Dptr, Dptr_host, sizeof(T) * M * N, cudaMemcpyHostToDevice);
  // 清零 cuBLAS 结果
  cudaMemset(Dptr_cublas, 0, sizeof(T) * M * N);
  cudaMemset(Dptr_cublaslt, 0, sizeof(T) * M * N);

  // 初始化 cuBLASLt GEMM
  CublasLtGemm<T, ComputeType> cublaslt_gemm;
  if (enable_cublaslt) {
    cublaslt_gemm.init(Dptr_cublaslt, Bptr, Aptr, N, M, K);
  }

  // 创建 GEMM 配置: FP16, 128x128x32 tile, 3 阶段流水线
  config::GemmConfig<T, 128, 128, 32, 3> gemm_config;

  // 计算 kernel 启动参数
  dim3 block = gemm_config.kThreadNum;
  dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
            (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
  int shm_size = gemm_config.kShmSize;

  // GEMM 的 alpha 和 beta 参数
  half alpha = 1.f;
  half beta = 0.f;

  // 打印配置信息
  printf("kThreadNum is %d\n", gemm_config.kThreadNum);
  printf("kMmaPM is %d\n", gemm_config.kMmaPM);
  printf("kMmaPN is %d\n", gemm_config.kMmaPN);
  printf("kMmaPK is %d\n", gemm_config.kMmaPK);

  // 运行多次测试
  for (int it = 0; it < nt; ++it) {
    // 运行 cuBLAS GEMM 作为基准
    cudaMemset(Dptr_cublas, 0, sizeof(T) * M * N);
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                                     &alpha, (half *)Bptr, K, (half *)Aptr, K,
                                     &beta, (half *)Dptr_cublas, N);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("cublas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }

    // 运行 cuBLASLt GEMM
    if (enable_cublaslt) {
      cudaMemset(Dptr_cublaslt, 0, sizeof(T) * M * N);
      cublaslt_gemm.run();
    }

    // 运行自定义的多阶段 GEMM kernel
    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    // 设置最大动态共享内存大小
    cudaFuncSetAttribute(gemm_multi_stage<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    // 启动 kernel
    gemm_multi_stage<decltype(gemm_config)>
        <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
  }

  // 将结果从设备复制回主机
  cudaMemcpy(Dptr_host, Dptr, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(Dptr_host_blas, Dptr_cublas, sizeof(T) * M * N,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(Dptr_host_cublaslt, Dptr_cublaslt, sizeof(T) * M * N,
             cudaMemcpyDeviceToHost);

  // 同步设备并检查错误
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("block = (%d, %d), gird = (%d, %d), shm = %d\n", block.x, block.y,
         grid.x, grid.y, shm_size);

  if (err == cudaSuccess) {
    printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
  } else {
    printf_fail("err = %d, str = %s\n", err, cudaGetErrorString(err));
  }

  // 比较自定义 GEMM 和 cuBLAS 的结果
  gpu_compare(Dptr, Dptr_cublas, M * N);

  // 比较自定义 GEMM 和 cuBLASLt 的结果
  if (enable_cublaslt) {
    gpu_compare(Dptr, Dptr_cublaslt, M * N);
  }

  // 创建主机张量用于打印
  auto tD_host = make_tensor(Dptr_host, make_shape(M, N), make_stride(N, 1));
  auto tD_host_cpu =
      make_tensor(Dptr_host_cpu, make_shape(M, N), make_stride(N, 1));
  auto tD_host_blas =
      make_tensor(Dptr_host_blas, make_shape(M, N), make_stride(N, 1));
  auto tD_host_cublaslt =
      make_tensor(Dptr_host_cublaslt, make_shape(M, N), make_stride(N, 1));

  // 运行 CPU GEMM 作为验证
  if (enable_cpu) {
    cpu_gemm(&tD_host_cpu, tA, tB);
    cpu_compare(tD_host_cpu, tD_host, 0.1f);
  }

  // 创建小 tile 用于打印结果（最多 8x8）
  auto tile = make_tile(min(8, M), min(8, N));
  auto t32x32 = local_tile(tD_host, tile, make_coord(0, 0));
  auto t32x32_cpu = local_tile(tD_host_cpu, tile, make_coord(0, 0));
  auto t32x32_blas = local_tile(tD_host_blas, tile, make_coord(0, 0));
  auto t32x32_cublaslt = local_tile(tD_host_cublaslt, tile, make_coord(0, 0));

  printf("M = %d, N = %d, K = %d\n", M, N, K);

  // 打印自定义实现的结果
  printf("our-impl:\n");
  print_tensor(t32x32);
  
  // 打印 CPU 的结果
  if (enable_cpu) {
    printf("cpu:\n");
    print_tensor(t32x32_cpu);
  }
  
  // 打印 cuBLAS 的结果
  printf("cublas:\n");
  print_tensor(t32x32_blas);

  // 打印 cuBLASLt 的结果
  if (enable_cublaslt) {
    printf("cublaslt:\n");
    print_tensor(t32x32_cublaslt);
  }
}