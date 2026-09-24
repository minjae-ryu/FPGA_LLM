"""Independent HF/NumPy oracles and public CUDA forward failure contracts."""
import ctypes as ct
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest
from test_cuda import Error, library, GPU
from test_runtime import CALLBACK

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
pytestmark = GPU

@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    import export_smollm as exporter
    folder = tmp_path_factory.mktemp("cuda-forward")
    cfg = exporter.synthetic_config()
    exporter.export_synthetic(folder / "f32.bin", "f32")
    torch.set_num_threads(1)
    config = LlamaConfig(hidden_size=cfg.dim, intermediate_size=cfg.intermediate,
        num_hidden_layers=cfg.layers, num_attention_heads=cfg.heads,
        num_key_value_heads=cfg.kv_heads, vocab_size=cfg.vocab,
        max_position_embeddings=cfg.max_context, rope_theta=cfg.rope_theta,
        rms_norm_eps=cfg.rms_epsilon, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    hf = LlamaForCausalLM(config).float().eval()
    weights = {k:v.float() for k,v in exporter.synthetic_tensors(cfg).items()}
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    hf.load_state_dict(weights, strict=True)
    tokens = ((np.arange(300)*7+3) % cfg.vocab).astype(np.uint32)
    with torch.inference_mode():
        reference = hf(torch.tensor(tokens.astype(np.int64))[None]).logits[0].numpy()
    lib = library("libsmollm-cuda.so")
    lib.sm_cuda_prefill.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32), ct.c_size_t,
        ct.c_int, CALLBACK, ct.c_void_p, ct.POINTER(Error)]
    lib.sm_cuda_decode.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_int, CALLBACK, ct.c_void_p, ct.POINTER(Error)]
    host, model, error = ct.c_void_p(), ct.c_void_p(), Error()
    assert lib.sm_model_load(str(folder / "f32.bin").encode(),ct.byref(host),ct.byref(error)) == 0, error.message
    assert lib.sm_cuda_model_create(host,0,ct.byref(model),ct.byref(error)) == 0, error.message
    lib.sm_model_free(host)
    yield lib,model,cfg,tokens,reference,{k:v.numpy() for k,v in weights.items()},folder
    lib.sm_cuda_model_free(model)


def session(runtime, chunk=128, context=384):
    lib,model = runtime[:2]
    s,error = ct.c_void_p(),Error()
    assert lib.sm_cuda_session_create(model,1,context,chunk,ct.byref(s),ct.byref(error)) == 0,error.message
    return s


def forward(lib,s,tokens,mode=2,cb=None):
    rows,positions = [],[]
    @CALLBACK
    def receive(_ctx,pos,values,vocab):
        rows.append(np.ctypeslib.as_array(values,shape=(vocab,)).copy())
        positions.append(pos)
        return 0
    cb = receive if cb is None else cb
    array = np.asarray(tokens,dtype=np.uint32)
    error = Error()
    rc=lib.sm_cuda_prefill(s,array.ctypes.data_as(ct.POINTER(ct.c_uint32)),len(array),mode,cb,None,ct.byref(error))
    assert rc == 0,error.message
    return np.array(rows),positions


@pytest.mark.parametrize("chunk", [1,127,128,129])
def test_hf_chunks_reset_modes_and_independent_sessions(runtime,chunk):
    lib,_,cfg,tokens,reference = runtime[:5]
    a,b = session(runtime,chunk),session(runtime,chunk)
    try:
        actual,pos = forward(lib,a,tokens)
        assert pos == list(range(len(tokens)))
        np.testing.assert_allclose(actual,reference,atol=1e-3,rtol=1e-4)
        # Two sessions sharing weights must not share KV; future tokens are masked.
        prefix,_ = forward(lib,b,tokens[:31])
        np.testing.assert_allclose(prefix,reference[:31],atol=1e-3,rtol=1e-4)
        error = Error()
        assert lib.sm_cuda_session_reset(a,ct.byref(error)) == 0
        rows,pos=forward(lib,a,tokens[:129],mode=0)
        assert not len(rows) and not pos
        last,pos=forward(lib,a,tokens[129:257],mode=1)
        assert pos == [256]
        np.testing.assert_allclose(last,reference[256:257],atol=1e-3,rtol=1e-4)
        assert lib.sm_cuda_session_reset(a,ct.byref(error)) == 0
        at=0
        for count in (1,17,127,3,129,23):
            rows,pos=forward(lib,a,tokens[at:at+count])
            np.testing.assert_allclose(rows,reference[at:at+count],atol=1e-3,rtol=1e-4)
            assert pos == list(range(at,at+count))
            at+=count
        # Existing C scoring computes FP64 NLL (not Python inference scoring).
        lib.sm_nll.argtypes=[ct.c_void_p,ct.c_size_t,ct.c_uint32]
        lib.sm_nll.restype=ct.c_double
        nll=sum(lib.sm_nll(row.ctypes.data,cfg.vocab,int(t)) for row,t in zip(actual[:-1],tokens[1:]))
        ref_nll=sum(lib.sm_nll(row.ctypes.data,cfg.vocab,int(t)) for row,t in zip(reference[:-1],tokens[1:]))
        assert abs(nll-ref_nll)/299 <= 1e-4
    finally:
        lib.sm_cuda_session_free(a);lib.sm_cuda_session_free(b)


def test_invalid_inputs_callback_failure_and_reentry(runtime):
    lib,_,cfg,tokens = runtime[:4]
    s=session(runtime,chunk=4,context=8)
    e=Error()
    try:
        for values,mode in [([1,cfg.vocab],0),([1]*9,0),([1],3)]:
            array=np.array(values,dtype=np.uint32)
            assert lib.sm_cuda_prefill(s,array.ctypes.data_as(ct.POINTER(ct.c_uint32)),len(array),mode,CALLBACK(),None,ct.byref(e)) == -1
            assert lib.sm_cuda_session_position(s) == 0
        assert lib.sm_cuda_prefill(s,None,0,2,CALLBACK(),None,ct.byref(e)) == 0
        assert lib.sm_cuda_prefill(s,None,1,2,CALLBACK(),None,ct.byref(e)) == -1
        calls=[]; nested_results=[]
        @CALLBACK
        def fail(ctx,pos,data,vocab):
            calls.append(pos)
            nested=Error()
            nested_results.append((lib.sm_cuda_session_reset(s,ct.byref(nested)), bytes(nested.message)))
            nested_results.append((lib.sm_cuda_decode(s,0,0,CALLBACK(),None,ct.byref(nested)), bytes(nested.message)))
            return 1
        array=np.asarray(tokens[:5],dtype=np.uint32)
        assert lib.sm_cuda_prefill(s,array.ctypes.data_as(ct.POINTER(ct.c_uint32)),5,2,fail,None,ct.byref(e)) == -1
        assert calls == [0] and lib.sm_cuda_session_position(s) == 0
        assert len(nested_results)==2 and all(rc==-1 and b"busy" in msg for rc,msg in nested_results)
        assert lib.sm_cuda_decode(s,0,0,CALLBACK(),None,ct.byref(e)) == -1
        assert b"reset" in e.message
        assert lib.sm_cuda_session_reset(s,ct.byref(e)) == 0
        # NULL callback still projects and validates logits, then advances position.
        assert lib.sm_cuda_decode(s,int(tokens[0]),1,CALLBACK(),None,ct.byref(e)) == 0
        assert lib.sm_cuda_session_position(s) == 1
        forward(lib,s,tokens[1:8],mode=0)
        assert lib.sm_cuda_session_position(s) == 8
        assert lib.sm_cuda_decode(s,0,0,CALLBACK(),None,ct.byref(e)) == -1
    finally:
        lib.sm_cuda_session_free(s)


def read_trace(path):
    result={}
    with path.open("rb") as f:
        assert f.read(8)==b"SMTRC001"
        while header:=f.read(128):
            site=header[:64].split(b"\0")[0].decode()
            layer,head,pos=struct.unpack_from("<iiQ",header,64)
            ni,no=struct.unpack_from("<QQ",header,96)
            inp=np.frombuffer(f.read(ni*4),dtype="<f4").copy()
            out=np.frombuffer(f.read(no*4),dtype="<f4").copy()
            result[(site,layer,head,pos)]=(inp,out)
    return result


def test_independent_tensor_and_causal_attention_oracles(runtime):
    _,_,c,tokens,_,w,folder=runtime
    path=folder/"trace.bin"
    tp=folder/"tokens.bin"
    tp.write_bytes(struct.pack("<8sI",b"SMTOK001",7)+tokens[:7].tobytes())
    run=subprocess.run([ROOT/"build/smollm-cuda","logits","--backend","cuda","--model",folder/"f32.bin",
        "--tokens",tp,"--trace",path,"--chunk","4","--context","16"],capture_output=True,text=True,timeout=60)
    assert run.returncode == 0,run.stderr
    assert json.loads(run.stdout)["backend"] == "cuda"
    t=read_trace(path)
    for (site,layer,head,pos),(inp,out) in t.items():
        if site in w and len(inp):
            np.testing.assert_allclose(out,w[site]@inp,atol=2e-5,rtol=1e-4)
        if site in ("attn_norm","ffn_norm"):
            name="input_layernorm" if site=="attn_norm" else "post_attention_layernorm"
            expected=inp/np.sqrt(np.mean(inp*inp)+c.rms_epsilon)*w[f"model.layers.{layer}.{name}.weight"]
            np.testing.assert_allclose(out,expected,atol=2e-6,rtol=1e-5)
        if site=="embedding": np.testing.assert_array_equal(out,w["model.embed_tokens.weight"][tokens[pos]])
        if site in ("q_rope","k_rope"):
            kind="q" if site=="q_rope" else "k"
            before=t[(f"model.layers.{layer}.self_attn.{kind}_proj.weight",layer,-1,pos)][1].reshape(-1,c.head_dim)
            freq=1/np.power(np.float32(c.rope_theta),np.arange(0,c.head_dim,2,dtype=np.float32)/c.head_dim)
            sn,cs=np.sin(pos*freq),np.cos(pos*freq)
            left,right=before[:,:c.head_dim//2],before[:,c.head_dim//2:]
            expected=np.concatenate([left*cs-right*sn,right*cs+left*sn],axis=1).ravel()
            np.testing.assert_allclose(out,expected,atol=2e-6,rtol=1e-5)
        if site=="attention.probabilities":
            q=t[("q_rope",layer,-1,pos)][1].reshape(c.heads,c.head_dim)[head]
            kh=head//(c.heads//c.kv_heads)
            keys=np.array([t[("k_rope",layer,-1,j)][1].reshape(c.kv_heads,c.head_dim)[kh] for j in range(pos+1)])
            scores=keys@q/np.sqrt(np.float32(c.head_dim))
            probs=np.exp(scores-scores.max());probs/=probs.sum()
            np.testing.assert_allclose(out[:pos+1],probs,atol=2e-6,rtol=1e-5)
            np.testing.assert_array_equal(out[pos+1:],0)
        if site=="attention":
            expected=[]
            for h in range(c.heads):
                kh=h//(c.heads//c.kv_heads)
                probs=t[("attention.probabilities",layer,h,pos)][1][:pos+1]
                values=np.array([t[("v",layer,-1,j)][1].reshape(c.kv_heads,c.head_dim)[kh] for j in range(pos+1)])
                expected.extend(probs@values)
            np.testing.assert_allclose(out,expected,atol=2e-6,rtol=1e-5)
        if site=="gate_silu":
            g=t[(f"model.layers.{layer}.mlp.gate_proj.weight",layer,-1,pos)][1]
            np.testing.assert_allclose(out,g/(1+np.exp(-g)),atol=2e-6,rtol=1e-5)
