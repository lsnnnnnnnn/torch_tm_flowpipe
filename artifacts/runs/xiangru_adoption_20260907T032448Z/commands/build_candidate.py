import json
import os
from pathlib import Path
import sys
import time
import torch
import flowstar_gpu
from flowstar_gpu import cuda_kernels as ck

root = Path(__file__).resolve().parent
torch.set_num_threads(1)
start = time.perf_counter()
available = ck.available()
torch.cuda.synchronize()
record = dict(python_executable=sys.executable, python_version=sys.version,
              torch_version=torch.__version__, torch_file=torch.__file__,
              cuda_build=torch.version.cuda, imported_package=flowstar_gpu.__file__,
              import_paths=list(flowstar_gpu.__path__), extension_available=available,
              extension_path=ck._ext.__file__ if available else None,
              build_wall_seconds=time.perf_counter()-start,
              gpu_name=torch.cuda.get_device_name(0), gpu_capability=torch.cuda.get_device_capability(0),
              gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,
              threads=torch.get_num_threads(), affinity=sorted(os.sched_getaffinity(0)),
              build_environment={k:os.environ.get(k) for k in
                  ['CUDA_HOME','CUDA_VISIBLE_DEVICES','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR',
                   'TORCH_CUDA_ARCH_LIST','FLOWSTAR_CUDA_HOST_COMPILER','MAX_JOBS','CXX']})
(root/'candidate_build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2), flush=True)
if not available: raise RuntimeError('Extension build failed; production fallback is not accepted as a compiled-kernel run.')
