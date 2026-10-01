import hashlib
def _no_digest(*args, **kwargs):
    raise RuntimeError("content digest disabled")
_original_new = hashlib.new
def _guarded_new(name, *args, **kwargs):
    if name.lower().replace("-", "") == "sha256":
        return _no_digest()
    return _original_new(name, *args, **kwargs)
hashlib.sha256 = _no_digest
hashlib.new = _guarded_new
import os
import sys
CROWN_DIR = "../../Verifier_Development/complete_verifier"
sys.path.append(CROWN_DIR)

import torch
import onnx
import onnx2pytorch
from collections import defaultdict, OrderedDict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# For RPC
import gevent
import gevent.pywsgi
import gevent.queue
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.wsgi import WsgiServerTransport
from tinyrpc.server.gevent import RPCServerGreenlets
from tinyrpc.dispatch import RPCDispatcher


dispatcher = RPCDispatcher()
transport = WsgiServerTransport(max_content_length=200000, queue_class=gevent.queue.Queue)
# start wsgi server as a background-greenlet
wsgi_server = gevent.pywsgi.WSGIServer(('127.0.0.1', 5101), transport.handle)
gevent.spawn(wsgi_server.serve_forever)
rpc_server = RPCServerGreenlets(transport, JSONRPCProtocol(), dispatcher)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
torch.set_default_dtype(torch.float64)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

benchmark_dir = "../../ARCH-COMP2024/benchmarks/Single_Pendulum/"
model_path = os.environ["SP_MODEL"]
assert os.path.isfile(model_path)
input_shape = (1, 2)
output_T_shape = (1, 2)
output_c_shape = (1)
output_scale = 1
output_offset = 0
onnx_model = onnx.load(model_path)
model_ori = onnx2pytorch.ConvertModel(onnx_model)
model_ori.to(torch.get_default_dtype())
lirpa_model = BoundedModule(model_ori, torch.zeros(*input_shape), device=device, bound_opts={'activation_bound_option': 'same-slope'})

@dispatcher.public
def CROWN_reach(input_lb, input_ub):
    input_lb = torch.tensor(input_lb).view(*input_shape).to(dtype=torch.get_default_dtype(), device=device)
    input_ub = torch.tensor(input_ub).view(*input_shape).to(dtype=torch.get_default_dtype(), device=device)
    ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
    bounded_tensor = BoundedTensor(input_lb, ptb)
    required_A = defaultdict(set)
    required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
    lb, ub, A_dict = lirpa_model.compute_bounds(x=(bounded_tensor,), method='CROWN', return_A=True, needed_A_dict=required_A)
    A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]
    coefficients = OrderedDict()
    coefficients["T"] = (A['lA'] * output_scale).view(output_T_shape).cpu().tolist()
    coefficients["u_min"] = ((A['lbias'] - output_offset) * output_scale).view(output_c_shape).cpu().tolist()
    coefficients["u_max"] = ((A['ubias'] - output_offset) * output_scale).view(output_c_shape).cpu().tolist()
    if os.environ.get('ARCH_RPC_LOG'):
        import json
        with open(os.environ['ARCH_RPC_LOG'], 'a') as log:
            log.write(json.dumps({'input_lower':input_lb.detach().cpu().reshape(-1).tolist(),'input_upper':input_ub.detach().cpu().reshape(-1).tolist(),'response':coefficients},allow_nan=False)+'\n')
    return coefficients

print("Server started.")
rpc_server.serve_forever()