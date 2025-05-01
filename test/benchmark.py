import torchmm
import torch
import time
from prettytable import PrettyTable

dtype_lst = [torch.int8, torch.int16, torch.int32, torch.int64]
batch_lst = [1, 4, 8, 16, 32, 64, 128, 256]
mat_shape = (256, 256)

average_cost_cpu = {}
average_cost_cuda = {}
total_test_num = len(batch_lst) * len(dtype_lst) * 2
current_test_n = 1
for dtype in dtype_lst:
    average_cost_cpu[dtype], average_cost_cuda[dtype] = [], []
    for batch in batch_lst:
        mat1 = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (batch, *mat_shape), dtype=dtype)
        mat2 = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (batch, *mat_shape), dtype=dtype)

        mat1_cpu, mat2_cpu = mat1.cpu(), mat2.cpu()
        mat1_cuda, mat2_cuda = mat1.cuda(), mat2.cuda()

        start_ns = time.time_ns()
        mat3_cpu = torch.matmul(mat1_cpu, mat2_cpu)
        stop_ns = time.time_ns()
        time_cost_ns_cpu = stop_ns - start_ns

        start_ns = time.time_ns()
        mat3_cuda = torchmm.matmul(mat1_cuda, mat2_cuda)
        torch.cuda.synchronize(torch.device("cuda"))
        stop_ns = time.time_ns()
        time_cost_ns_cuda = stop_ns - start_ns

        assert torch.equal(mat3_cuda.cpu(), mat3_cpu) is True, "Error!"
        print(f"{current_test_n:8d} / {total_test_num} "
              f"[cpu ] {dtype} {mat1_cpu.shape} x {mat2_cpu.shape}: {time_cost_ns_cpu / 1e6:.6f} ms")
        current_test_n += 1
        print(f"{current_test_n:8d} / {total_test_num} "
              f"[cuda] {dtype} {mat1_cuda.shape} x {mat2_cuda.shape}: {time_cost_ns_cuda / 1e6:.6f} ms")
        current_test_n += 1

        average_cost_cpu[dtype].append(time_cost_ns_cpu / batch / 1e6)
        average_cost_cuda[dtype].append(time_cost_ns_cuda / batch / 1e6)


for dtype in dtype_lst:
    tb = PrettyTable()
    tb.add_column("shape", [f"{mat_shape} x {mat_shape}"] * len(batch_lst))
    tb.add_column("dtype", [dtype] * len(batch_lst))
    tb.add_column("batchSize", batch_lst)
    tb.add_column("cpuSpeed(ms/op)", average_cost_cpu[dtype])
    tb.add_column("cudaSpeed(ms/op)", average_cost_cuda[dtype])
    tb.add_column("speedUp", [
        "{:g}".format(average_cost_cpu[dtype][i] / (average_cost_cuda[dtype][i] + 1e-9)) for i in range(len(batch_lst))
    ])
    print(tb)
