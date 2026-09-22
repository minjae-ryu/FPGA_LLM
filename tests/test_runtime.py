"""Independent runtime checks through the public C ABI and HF eager forward."""
import ctypes as ct
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_smollm as exporter
from smollm_format import encode_q8_groups, parse_model

class Error(ct.Structure):
    _fields_ = [("message", ct.c_char * 256)]

CALLBACK = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_float), ct.c_size_t)

@pytest.fixture(scope="module")
def lib():
    library = ct.CDLL(str(ROOT / "build/libsmollm.so"))
    library.sm_model_load.argtypes = [ct.c_char_p, ct.POINTER(ct.c_void_p), ct.POINTER(Error)]
    library.sm_model_free.argtypes = [ct.c_void_p]
    library.sm_session_create.argtypes = [ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_size_t, ct.POINTER(ct.c_void_p), ct.POINTER(Error)]
    library.sm_session_free.argtypes = [ct.c_void_p]
    library.sm_session_reset.argtypes = [ct.c_void_p]
    library.sm_session_position.argtypes = [ct.c_void_p]
    library.sm_session_position.restype = ct.c_size_t
    library.sm_session_kv_bytes.argtypes = [ct.c_void_p]
    library.sm_session_kv_bytes.restype = ct.c_size_t
    library.sm_prefill.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32), ct.c_size_t, ct.c_int, CALLBACK, ct.c_void_p, ct.POINTER(Error)]
    library.sm_quantize.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.POINTER(Error)]
    library.sm_dequantize.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
    library.sm_nll.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_uint32]
    library.sm_nll.restype = ct.c_double
    return library

@pytest.fixture(scope="module")
def models(tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    for dtype in ("f32", "q8"):
        exporter.export_synthetic(folder / f"{dtype}.bin", dtype)
    return folder

def load(lib, path):
    error, model = Error(), ct.c_void_p()
    result = lib.sm_model_load(str(path).encode(), ct.byref(model), ct.byref(error))
    assert result == 0, error.message.decode()
    return model

def session(lib, model, kv=1, chunk=128, context=384):
    error, state = Error(), ct.c_void_p()
    result = lib.sm_session_create(model, kv, context, chunk, ct.byref(state), ct.byref(error))
    assert result == 0, error.message.decode()
    return state

def forward(lib, state, tokens, mode=2):
    rows, positions = [], []
    @CALLBACK
    def receive(_ctx, pos, values, width):
        rows.append(np.ctypeslib.as_array(values, shape=(width,)).copy())
        positions.append(pos)
        return 0
    array = np.asarray(tokens, dtype=np.uint32)
    error = Error()
    result = lib.sm_prefill(state, array.ctypes.data_as(ct.POINTER(ct.c_uint32)), len(array), mode, receive, None, ct.byref(error))
    assert result == 0, error.message.decode()
    return np.array(rows), positions

def test_fp32_independent_hf_and_causal_prefix(lib, models):
    torch.set_num_threads(1)
    cfg = exporter.synthetic_config()
    hf_config = LlamaConfig(hidden_size=cfg.dim, intermediate_size=cfg.intermediate,
        num_hidden_layers=cfg.layers, num_attention_heads=cfg.heads,
        num_key_value_heads=cfg.kv_heads, vocab_size=cfg.vocab,
        max_position_embeddings=cfg.max_context, rope_theta=cfg.rope_theta,
        rms_norm_eps=cfg.rms_epsilon, tie_word_embeddings=True)
    hf_config._attn_implementation = "eager"
    hf = LlamaForCausalLM(hf_config).float().eval()
    state = {k: v.float() for k, v in exporter.synthetic_tensors(cfg).items()}
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    hf.load_state_dict(state, strict=True)
    tokens = (np.arange(300) * 7 + 3) % cfg.vocab
    with torch.inference_mode():
        reference = hf(torch.tensor(tokens).unsqueeze(0)).logits[0].numpy()
    model = load(lib, models / "f32.bin")
    try:
        state = session(lib, model)
        try:
            actual, positions = forward(lib, state, tokens)
            assert positions == list(range(300))
            np.testing.assert_allclose(actual, reference, atol=1e-3, rtol=1e-4)
            smoothed = []
            for logits, target in zip(actual[:-1], tokens[1:]):
                smoothed.append(lib.sm_nll(logits.ctypes.data, cfg.vocab, int(target)))
            ref64 = reference[:-1].astype(np.float64)
            maxes = ref64.max(axis=1)
            expected = maxes + np.log(np.exp(ref64-maxes[:,None]).sum(axis=1)) - ref64[np.arange(299),tokens[1:]]
            assert abs(np.mean(smoothed)-expected.mean()) <= 1e-4
            lib.sm_session_reset(state)
            prefix, _ = forward(lib, state, tokens[:31])
            np.testing.assert_allclose(prefix, actual[:31], atol=1e-3, rtol=1e-4)
        finally:
            lib.sm_session_free(state)
    finally:
        lib.sm_model_free(model)

@pytest.mark.parametrize("dtype,kv", [("f32",1),("q8",1),("f32",2),("q8",2)])
def test_chunk_reset_modes_bounds_and_cache_accounting(lib, models, dtype, kv):
    model = load(lib, models / f"{dtype}.bin")
    cfg = exporter.synthetic_config()
    tokens = (np.arange(300) * 11 + 1) % cfg.vocab
    baseline = None
    try:
        for chunk in (1,127,128,129):
            state = session(lib, model, kv, chunk)
            try:
                output, _ = forward(lib, state, tokens)
                if baseline is None:
                    baseline = output
                elif dtype == "f32" and kv == 1:
                    np.testing.assert_allclose(output, baseline, atol=1e-3, rtol=1e-4)
                # An independent trace review located Q8 differences at exact
                # quantizer discontinuities after sub-1e-5 BLAS differences.
                # There is no arbitrary Q8 accuracy gate. For each configuration
                # require deterministic reset and exact identical-call behavior;
                # fixed-input integer/cache parity is tested independently.
                assert np.isfinite(output).all()
                lib.sm_session_reset(state)
                reset_output, _ = forward(lib, state, tokens)
                np.testing.assert_array_equal(reset_output, output)
                lib.sm_session_reset(state)
                forward(lib, state, tokens[:129], mode=2)
                expected_last, _ = forward(lib, state, tokens[129:257], mode=2)
                lib.sm_session_reset(state)
                no_rows, no_positions = forward(lib, state, tokens[:129], mode=0)
                assert len(no_rows) == len(no_positions) == 0
                last, positions = forward(lib, state, tokens[129:257], mode=1)
                assert positions == [256]
                # LAST uses a GEMV output projection instead of ALL's GEMM.
                np.testing.assert_allclose(last[0], expected_last[-1], atol=1e-3, rtol=1e-4)
                position = lib.sm_session_position(state)
                error = Error()
                bad = (ct.c_uint32 * 1)(cfg.vocab)
                callback = CALLBACK(lambda *_: 0)
                assert lib.sm_prefill(state,bad,1,0,callback,None,ct.byref(error)) != 0
                assert lib.sm_session_position(state) == position
                overflow = np.zeros(384-position+1,dtype=np.uint32)
                assert lib.sm_prefill(state,overflow.ctypes.data_as(ct.POINTER(ct.c_uint32)),len(overflow),0,callback,None,ct.byref(error)) != 0
                assert lib.sm_session_position(state) == position
                forward(lib,state,np.zeros(384-position,dtype=np.uint32),mode=0)
                assert lib.sm_session_position(state) == 384
                kv_bytes = 2*cfg.layers*384*cfg.kv_heads*(cfg.head_dim*4 if kv==1 else cfg.head_dim//64*68)
                assert lib.sm_session_kv_bytes(state) == kv_bytes
            finally:
                lib.sm_session_free(state)
    finally:
        lib.sm_model_free(model)

@pytest.mark.parametrize("mutation", ["magic","endian","header","reserved","rank","dtype","shape","offset","bytes","overlap","duplicate","missing","nonfinite","truncated"])
def test_loader_rejects_corruption(lib, models, tmp_path, mutation):
    data = bytearray((models / "f32.bin").read_bytes())
    if mutation=="magic": data[0]=0
    elif mutation=="endian": struct.pack_into("<I",data,12,0x04030201)
    elif mutation=="header": struct.pack_into("<I",data,16,128)
    elif mutation=="reserved": data[112]=1
    elif mutation=="rank": struct.pack_into("<I",data,256+68,3)
    elif mutation=="dtype": struct.pack_into("<I",data,256+64,7)
    elif mutation=="shape": struct.pack_into("<Q",data,256+72,2**64-1)
    elif mutation=="offset": struct.pack_into("<Q",data,256+104,len(data)+64)
    elif mutation=="bytes": struct.pack_into("<Q",data,256+112,2**64-1)
    elif mutation=="overlap": data[384+104:384+112]=data[256+104:256+112]
    elif mutation=="duplicate": data[384:384+64]=data[256:256+64]
    elif mutation=="missing": data[256:256+64]=b"unknown\0"+bytes(56)
    elif mutation=="nonfinite": struct.pack_into("<I",data,struct.unpack_from("<Q",data,256+104)[0],0x7fc00000)
    elif mutation=="truncated": data=data[:-512]
    path=tmp_path/"bad.bin";path.write_bytes(data)
    model,error=ct.c_void_p(),Error()
    assert lib.sm_model_load(str(path).encode(),ct.byref(model),ct.byref(error)) != 0
    assert not model.value
    assert error.message

def test_c_quantization_exact_fixtures_and_random(lib):
    # Expected bytes come from the separately specified fixture and Python path.
    cases=json.loads((ROOT/"tests/fixtures/quantization.json").read_text())
    from test_export import _fixture_values
    for case in cases["valid"]:
        values=np.ascontiguousarray(_fixture_values(case).reshape(-1),dtype=np.float32)
        out=ct.create_string_buffer(values.size//64*68);error=Error()
        assert lib.sm_quantize(values.ctypes.data,out,values.size,ct.byref(error))==0
        assert out.raw==encode_q8_groups(values.reshape(-1,64))
    for case in cases["invalid"]:
        values=np.ascontiguousarray(_fixture_values(case).reshape(-1),dtype=np.float32)
        out=ct.create_string_buffer(max(68,values.size//64*68));error=Error()
        assert lib.sm_quantize(values.ctypes.data,out,values.size,ct.byref(error)) != 0
    values=np.random.default_rng(721).normal(size=(127,64)).astype(np.float32)
    out=ct.create_string_buffer(values.size//64*68);error=Error()
    assert lib.sm_quantize(values.ctypes.data,out,values.size,ct.byref(error))==0
    assert out.raw==encode_q8_groups(values)

def test_nll_uses_stable_fp64(lib):
    row=np.array([-1000,1000,999],dtype=np.float32)
    result=lib.sm_nll(row.ctypes.data,3,2)
    assert abs(result-(1+np.log1p(np.exp(-1))))<1e-12
