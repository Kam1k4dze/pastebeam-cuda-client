# pastebeam CUDA Client

A high-performance CUDA client for [pastebeam](https://github.com/tsoding/pastebeam), leveraging GPU acceleration to
speed up uploads.

## Prerequisites

- CUDA
- Boost

```bash
git clone https://github.com/tsoding/pastebeam-cuda-client.git
cd pastebeam-cuda-client

nvcc pastebeam_client.cu \
  -std=c++20 \
  -O3 \
  -Xcompiler "-march=native -O3" \
  -use_fast_math \
  -Xptxas="-O3,-v" \
  -arch=native \
  --relocatable-device-code=false \
  -o pastebeam_client
```

## Usage

Once compiled, the client supports two primary commands: `post` (upload) and `get` (download).

1. Upload a file
   ```bash
   ./pastebeam_client post path/to/file.txt
   ```  

2. Retrieve a paste by its hash
   ```bash
   ./pastebeam_client get C62872F9E0E9C575546166CE4ACB528A46E54BA4ED8AF076E79C0D5AD61C974B
   ```  

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
