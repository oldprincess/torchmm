# torchmm

为PyTorch整数类型开发的矩阵乘法GPU加速扩展，速度提高50倍。

A CUDA matrix multiplication extension for integer types for PyTorch, with a 50x speed up.

## Example

```python
import torch
import torchmm

mat1 = torch.randint(0, 256, (128, 128), device="cuda", dtype=torch.int32)
mat2 = torch.randint(0, 256, (128, 128), device="cuda", dtype=torch.int32)
print(torchmm.matmul(mat1, mat2))
```

## Installation

```shell
git clone --recursive https://github.com/oldprincess/torchmm.git --depth 1
```

```shell
python setup.py bdist_wheel
```

```shell
# depend on your python version and os platform
pip install dist/torchmm-1.0.0-cp39-cp39-win_amd64.whl
```

## BanchMark

- OS：Windows 11
- Python：3.9
- NVCC Version: 12.6
- CPU：i5-12500H
- GPU：RTX 3050

```text
+-------------------------+-------------+-----------+-----------------+------------------+---------+
|          shape          |    dtype    | batchSize | cpuSpeed(ms/op) | cudaSpeed(ms/op) | speedUp |
+-------------------------+-------------+-----------+-----------------+------------------+---------+
| (256, 256) x (256, 256) | torch.int64 |     1     |     25.0042     |      2.8859      | 8.66426 |
| (256, 256) x (256, 256) | torch.int64 |     4     |    25.759825    |     0.74595      | 34.5329 |
| (256, 256) x (256, 256) | torch.int64 |     8     |    26.3211125   |    0.6243625     | 42.1568 |
| (256, 256) x (256, 256) | torch.int64 |     16    |    27.1207125   |    0.5632625     | 48.1493 |
| (256, 256) x (256, 256) | torch.int64 |     32    |    25.9527375   |    0.49336875    | 52.6031 |
| (256, 256) x (256, 256) | torch.int64 |     64    |  26.3745234375  |   0.459259375    | 57.4284 |
| (256, 256) x (256, 256) | torch.int64 |    128    |  27.35897109375 |  0.44845859375   | 61.0067 |
| (256, 256) x (256, 256) | torch.int64 |    256    |  26.89751171875 |  0.500232421875  |  53.77  |
+-------------------------+-------------+-----------+-----------------+------------------+---------+
```

## Notes

本软件是AS IS的( 不提供任何保证， ( 不管是显式的还是隐式的，包括但不限于适销性保证、适用性保证、非侵权性保证 ) ) ，在任何情况下， ( 对于任何的权益追索、损失赔偿或者任何追责 ) ，作者或者版权所有人都不会负责。( 无论这些追责产生自合同、侵权，还是直接或间接来自于本软件以及与本软件使用或经营有关的情形 )

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
