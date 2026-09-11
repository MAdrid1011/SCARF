#!/usr/bin/env python3
"""Run bundled ScarfTop RTL with Verilator and export timing events.

This adapter compiles the checked-in monolithic RTL and an AXI-memory harness,
runs every supplied packed stimulus for all FSDR/SAES variants, and emits
``scarf-rtl-simulator-events-v2`` for ``export_claim_timing.py``. It fails if
the RTL does not reach ``io_done`` or the bound payload is structurally
invalid; it never substitutes analytic cycles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rtl_payload import (
    ParsedPayload,
    constant_claim_roles,
    parse_payload,
    validate_claim_payload,
    workload_descriptor,
)


INPUT_SCHEMA = "scarf-rtl-stimulus-v1"
EVENT_SCHEMA = "scarf-rtl-simulator-events-v2"
VARIANTS = ("asic", "asic_fsdr", "asic_saes", "asic_fsdr_saes")


def tensor_data_image(payload: ParsedPayload) -> bytes:
    """Return only the packed tensor data for the AXI payload window."""
    data_size = sum(tensor.size_bytes for tensor in payload.tensors)
    with payload.path.open("rb") as stream:
        stream.seek(payload.data_offset_bytes)
        image = stream.read(data_size)
    if len(image) != data_size:
        raise ValueError("payload tensor-data section is truncated")
    return image


def consumption_evidence(
    descriptor: dict[str, Any], image: bytes, addresses: list[int]
) -> dict[str, Any]:
    """Bind observed aligned AXI reads to declared tensor roles and bytes."""
    ranges = descriptor.get("tensor_ranges")
    if not isinstance(ranges, list):
        raise ValueError("workload descriptor has no tensor ranges")
    coverage: dict[str, dict[str, int]] = {}
    digest = hashlib.sha256()
    for address in addresses:
        if not isinstance(address, int) or address < 0x90000000:
            raise ValueError("AXI consumption address is invalid")
        offset = address - 0x90000000
        if offset < 0 or offset >= len(image):
            raise ValueError("AXI consumption address is outside tensor data")
        data = image[offset : offset + 16]
        digest.update(address.to_bytes(8, "little", signed=False))
        digest.update(len(data).to_bytes(2, "little", signed=False))
        digest.update(data)
        for entry in ranges:
            role = entry.get("role") if isinstance(entry, dict) else None
            start = entry.get("offset_bytes") if isinstance(entry, dict) else None
            end = entry.get("end_bytes") if isinstance(entry, dict) else None
            if not isinstance(role, str) or not isinstance(start, int) or not isinstance(end, int):
                continue
            overlap = max(0, min(offset + len(data), end) - max(offset, start))
            if overlap:
                current = coverage.setdefault(role, {"bytes_read": 0, "range_count": 0})
                current["bytes_read"] += overlap
                current["range_count"] += 1
    return {
        "role_coverage": coverage,
        "axi_consumption_digest": digest.hexdigest(),
    }

HARNESS = r'''#include "VScarfTop.h"
#include "verilated.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
static std::vector<unsigned char> load_payload(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open payload: " + path);
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(stream), {});
}
static unsigned long long u64_arg(const char* value, const char* label) {
  try { return std::stoull(value); }
  catch (...) { throw std::runtime_error(std::string("invalid ") + label); }
}
static void drive_payload(VScarfTop* t, const std::vector<unsigned char>& payload,
                          std::vector<unsigned long long>& reads) {
  t->io_axiRData[0]=0; t->io_axiRData[1]=0; t->io_axiRData[2]=0; t->io_axiRData[3]=0;
  if (payload.empty()) return;
  // Keep the payload window disjoint from the normal weight/configuration
  // reads at 0x80000000 so the evidence log cannot count unrelated traffic.
  const unsigned long long base = 0x90000000ULL;
  const unsigned long long address = static_cast<unsigned long long>(t->io_axiArAddr);
  if (t->io_axiArValid && t->io_axiArReady && address >= base &&
      address < base + payload.size())
    reads.push_back(address);
  if (address < base || address >= base + payload.size()) return;
  const unsigned long long offset = address - base;
  for (unsigned i=0; i<16; ++i) {
    const unsigned long long index = offset + i;
    if (index >= payload.size()) continue;
    const unsigned word = i / 4;
    const unsigned shift = (i % 4) * 8;
    t->io_axiRData[word] |= static_cast<unsigned>(payload[index]) << shift;
  }
}
static void tick(VScarfTop* t, const std::vector<unsigned char>& payload,
                 std::vector<unsigned long long>& reads) {
  // Evaluate the falling edge first, then drive the response for exactly one
  // rising-edge handshake.  This prevents the same AR request being logged
  // once at each clock level.
  t->clock=0; t->eval(); drive_payload(t, payload, reads);
  t->clock=1; t->eval();
}
static void defaults(VScarfTop* t) {
  t->io_cfgReadAddr=0; t->io_cfgWriteAddr=0; t->io_cfgWriteData=0; t->io_cfgWriteEn=0;
  t->io_start=0; t->io_axiArReady=1; t->io_axiRValid=1; t->io_axiRLast=1;
  t->io_axiRData[0]=0; t->io_axiRData[1]=0; t->io_axiRData[2]=0; t->io_axiRData[3]=0;
  t->io_axiAwReady=1; t->io_axiWReady=1;
}
static void cfg(VScarfTop* t, unsigned a, unsigned d, const std::vector<unsigned char>& payload,
                std::vector<unsigned long long>& reads) {
  t->io_cfgWriteAddr=a; t->io_cfgWriteData=d; t->io_cfgWriteEn=1; tick(t,payload,reads); t->io_cfgWriteEn=0; tick(t,payload,reads);
}
static int stage(unsigned s) { if(s>=2&&s<=4)return 1; if(s>=5&&s<=11)return 2; if(s>=12&&s<=13)return 3; if(s==14)return 4; return 0; }
int main(int argc,char** argv) {
  bool saes=false, fsdr=false, limit_enabled=false; unsigned long long limit=0;
  std::string payload_path;
  unsigned image_h=256,image_w=256,feature_dim=256,num_depth=64,tensor_count=1;
  unsigned tile_size=4,cnn_layers=6,transformer_layers=6,norm_groups=8,sh_degree=4;
  unsigned has_dinov2=0,dinov2_layers=0,saes_feature_var=0,saes_cross_check=0,saes_depth_std=0;
  unsigned fsdr_cache_size=32,fsdr_hamming_threshold=3,fsdr_depth_valid_threshold=102;
  unsigned long long num_gaussians=1,payload_bytes=0,feature_offset=0,depth_offset=0,candidate_offset=0,probability_offset=0,saes_route_offset=0,feature_bytes=0,depth_bytes=0,candidate_bytes=0,probability_bytes=0,saes_route_bytes=0;
  for(int i=1;i<argc;i++){std::string a(argv[i]); if(a=="--saes")saes=true; else if(a=="--fsdr")fsdr=true; else if(a=="--payload"&&i+1<argc)payload_path=argv[++i]; else if(a=="--max-cycles"&&i+1<argc){limit_enabled=true;limit=u64_arg(argv[++i],"max-cycles");}
    else if(a=="--image-h"&&i+1<argc)image_h=std::strtoul(argv[++i],nullptr,10); else if(a=="--image-w"&&i+1<argc)image_w=std::strtoul(argv[++i],nullptr,10);
    else if(a=="--feature-dim"&&i+1<argc)feature_dim=std::strtoul(argv[++i],nullptr,10); else if(a=="--num-depth"&&i+1<argc)num_depth=std::strtoul(argv[++i],nullptr,10);
    else if(a=="--tile-size"&&i+1<argc)tile_size=std::strtoul(argv[++i],nullptr,10); else if(a=="--cnn-layers"&&i+1<argc)cnn_layers=std::strtoul(argv[++i],nullptr,10);
    else if(a=="--transformer-layers"&&i+1<argc)transformer_layers=std::strtoul(argv[++i],nullptr,10); else if(a=="--norm-groups"&&i+1<argc)norm_groups=std::strtoul(argv[++i],nullptr,10);
    else if(a=="--sh-degree"&&i+1<argc)sh_degree=std::strtoul(argv[++i],nullptr,10); else if(a=="--has-dinov2"&&i+1<argc)has_dinov2=std::strtoul(argv[++i],nullptr,10); else if(a=="--dinov2-layers"&&i+1<argc)dinov2_layers=std::strtoul(argv[++i],nullptr,10);
    else if(a=="--saes-feature-var"&&i+1<argc)saes_feature_var=std::strtoul(argv[++i],nullptr,10); else if(a=="--saes-cross-check"&&i+1<argc)saes_cross_check=std::strtoul(argv[++i],nullptr,10);
    else if(a=="--saes-depth-std"&&i+1<argc)saes_depth_std=std::strtoul(argv[++i],nullptr,10); else if(a=="--fsdr-cache-size"&&i+1<argc)fsdr_cache_size=std::strtoul(argv[++i],nullptr,10);
    else if(a=="--fsdr-hamming-threshold"&&i+1<argc)fsdr_hamming_threshold=std::strtoul(argv[++i],nullptr,10); else if(a=="--fsdr-depth-valid-threshold"&&i+1<argc)fsdr_depth_valid_threshold=std::strtoul(argv[++i],nullptr,10);
    else if(a=="--num-gaussians"&&i+1<argc)num_gaussians=u64_arg(argv[++i],"num-gaussians"); else if(a=="--payload-bytes"&&i+1<argc)payload_bytes=u64_arg(argv[++i],"payload-bytes");
    else if(a=="--payload-tensors"&&i+1<argc)tensor_count=std::strtoul(argv[++i],nullptr,10); else if(a=="--feature-offset"&&i+1<argc)feature_offset=u64_arg(argv[++i],"feature-offset"); else if(a=="--depth-offset"&&i+1<argc)depth_offset=u64_arg(argv[++i],"depth-offset"); else if(a=="--candidate-offset"&&i+1<argc)candidate_offset=u64_arg(argv[++i],"candidate-offset"); else if(a=="--probability-offset"&&i+1<argc)probability_offset=u64_arg(argv[++i],"probability-offset"); else if(a=="--saes-route-offset"&&i+1<argc)saes_route_offset=u64_arg(argv[++i],"saes-route-offset"); else if(a=="--feature-bytes"&&i+1<argc)feature_bytes=u64_arg(argv[++i],"feature-bytes"); else if(a=="--depth-bytes"&&i+1<argc)depth_bytes=u64_arg(argv[++i],"depth-bytes"); else if(a=="--candidate-bytes"&&i+1<argc)candidate_bytes=u64_arg(argv[++i],"candidate-bytes"); else if(a=="--probability-bytes"&&i+1<argc)probability_bytes=u64_arg(argv[++i],"probability-bytes"); else if(a=="--saes-route-bytes"&&i+1<argc)saes_route_bytes=u64_arg(argv[++i],"saes-route-bytes"); }
  if(image_h==0||image_w==0||feature_dim==0||num_depth==0||num_gaussians==0||payload_bytes==0||tensor_count==0||feature_bytes==0||depth_bytes==0||candidate_bytes==0||probability_bytes==0) throw std::runtime_error("workload descriptor contains zero");
  std::vector<unsigned char> payload;
  try { if(!payload_path.empty()) payload=load_payload(payload_path); }
  catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 4;}
  std::vector<unsigned long long> reads;
  Verilated::commandArgs(argc,argv); auto* t=new VScarfTop; defaults(t); t->reset=1; for(int i=0;i<8;i++)tick(t,payload,reads); t->reset=0;
  cfg(t,0x00,num_depth,payload,reads); cfg(t,0x04,feature_dim,payload,reads); cfg(t,0x08,image_h,payload,reads); cfg(t,0x0c,image_w,payload,reads); cfg(t,0x10,tile_size,payload,reads); cfg(t,0x14,cnn_layers,payload,reads); cfg(t,0x18,transformer_layers,payload,reads); cfg(t,0x1c,norm_groups,payload,reads); cfg(t,0x20,sh_degree,payload,reads); cfg(t,0x24,has_dinov2,payload,reads); cfg(t,0x28,saes_feature_var,payload,reads); cfg(t,0x2c,saes_cross_check,payload,reads); cfg(t,0x30,saes_depth_std,payload,reads); cfg(t,0x34,saes?1:0,payload,reads); cfg(t,0x38,fsdr?1:0,payload,reads); cfg(t,0x3c,fsdr_cache_size,payload,reads); cfg(t,0x40,fsdr_hamming_threshold,payload,reads); cfg(t,0x48,fsdr_depth_valid_threshold,payload,reads);
  cfg(t,0x4c,0x90000000U,payload,reads); cfg(t,0x50,static_cast<unsigned>(payload_bytes),payload,reads); cfg(t,0x54,tensor_count,payload,reads); cfg(t,0x58,static_cast<unsigned>(num_gaussians & 0xffffffffULL),payload,reads); cfg(t,0x60,1,payload,reads); cfg(t,0x64,static_cast<unsigned>(feature_offset),payload,reads); cfg(t,0x68,static_cast<unsigned>(depth_offset),payload,reads); cfg(t,0x6c,static_cast<unsigned>(probability_offset),payload,reads); cfg(t,0x70,static_cast<unsigned>(candidate_offset),payload,reads); cfg(t,0x74,static_cast<unsigned>(feature_bytes),payload,reads); cfg(t,0x78,static_cast<unsigned>(depth_bytes),payload,reads); cfg(t,0x7c,static_cast<unsigned>(candidate_bytes),payload,reads); cfg(t,0x80,static_cast<unsigned>(probability_bytes),payload,reads); cfg(t,0x84,dinov2_layers,payload,reads); cfg(t,0x88,static_cast<unsigned>(saes_route_offset),payload,reads); cfg(t,0x8c,static_cast<unsigned>(saes_route_bytes),payload,reads); cfg(t,0x44,1,payload,reads);
  unsigned stageCycles[5]={0,0,0,0,0}; unsigned c=0; t->io_start=1;
  for(;!limit_enabled||c<limit;c++){int now=stage(t->io_pipeState); if(now>0)stageCycles[now]++; if(t->io_done)break; tick(t,payload,reads);}
  bool ok=t->io_done&&(!limit_enabled||c<limit); if(!ok){std::cerr<<"RTL did not reach io_done";if(limit_enabled)std::cerr<<" before max_cycles="<<limit;std::cerr<<" state="<<unsigned(t->io_pipeState)<<" busy="<<unsigned(t->io_busy)<<" arvalid="<<unsigned(t->io_axiArValid)<<" rready="<<unsigned(t->io_axiRReady)<<"\n";return 2;}
  for(int i=1;i<=4;i++)if(stageCycles[i]==0){std::cerr<<"missing stage event "<<i<<"\n";return 3;}
  // A workload may revisit S2/S3 for multiple tiles.  Events therefore use
  // accumulated per-stage occupancy, laid out contiguously for the strict
  // trace schema; total_cycles retains the complete wall-clock duration.
  const char* n[]={"","s1","s2","s3","s4"}; unsigned cursor=0; std::cout<<"{\"total_cycles\":"<<(c+1)<<",\"events\":[";
  for(int i=1;i<=4;i++){if(i>1)std::cout<<",";std::cout<<"{\"stage\":\""<<n[i]<<"\",\"start_cycle\":"<<cursor<<",\"end_cycle\":"<<(cursor+stageCycles[i])<<",\"accepted\":true}"; cursor += stageCycles[i];} std::cout<<"],\"axi_read_count\":"<<reads.size()<<",\"axi_read_addresses\":[";
  for(size_t i=0;i<reads.size();i++){if(i)std::cout<<",";std::cout<<reads[i];} std::cout<<"],\"mechanism_counters\":{\"fsdr_narrow\":"<<t->io_fsdrNarrowCount<<",\"fsdr_full\":"<<t->io_fsdrFullCount<<",\"saes_l0\":"<<t->io_saesL0Count<<",\"saes_l1\":"<<t->io_saesL1Count<<",\"saes_full\":"<<t->io_saesFullCount<<"},\"workload\":{\"image_h\":"<<image_h<<",\"image_w\":"<<image_w<<",\"feature_dim\":"<<feature_dim<<",\"num_depth_candidates\":"<<num_depth<<",\"num_gaussians\":"<<num_gaussians<<",\"payload_bytes\":"<<payload_bytes<<",\"payload_tensor_count\":"<<tensor_count<<",\"tile_size\":"<<tile_size<<",\"cnn_layers\":"<<cnn_layers<<",\"transformer_layers\":"<<transformer_layers<<",\"norm_groups\":"<<norm_groups<<",\"sh_degree\":"<<sh_degree<<",\"has_dinov2\":"<<has_dinov2<<",\"dinov2_layers\":"<<dinov2_layers<<"}}\n"; delete t; return 0;
}
'''


def _stimuli(root: Path) -> list[tuple[dict[str, Any], Path]]:
    values: list[tuple[dict[str, Any], Path]] = []
    for path in sorted(root.resolve().rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("schema_version") == INPUT_SCHEMA:
            values.append((value, path))
    if not values:
        raise ValueError(f"no {INPUT_SCHEMA} files found below {root}")
    return values


def _compile(source: Path, build: Path) -> Path:
    verilator = shutil.which("verilator")
    if verilator is None:
        raise FileNotFoundError("verilator is required")
    harness = build / "scarf_timing_harness.cpp"
    harness.write_text(HARNESS, encoding="utf-8")
    mdir = build / "obj_dir"
    subprocess.run([verilator,"--cc",str(source),"--top-module","ScarfTop","--exe",str(harness),"--build","-Mdir",str(mdir),"-CFLAGS","-std=c++17","-Wno-fatal"], cwd=build, check=True)
    executable = mdir / "VScarfTop"
    if not executable.is_file():
        raise FileNotFoundError(f"missing Verilator executable: {executable}")
    return executable


def _payload_for(
    item: dict[str, Any], stimulus_path: Path, root: Path, *, allow_empty: bool
) -> tuple[Path | None, str | None, int, dict[str, Any] | None, ParsedPayload | None]:
    payload = item.get("payload")
    if not isinstance(payload, dict) or payload.get("schema_version") != "scarf-packed-payload-v1":
        if allow_empty:
            return None, None, 0, None, None
        raise ValueError(
            "stimulus has no packed payload; use --structural-only only for a smoke check"
        )
    relative = payload.get("path")
    digest = payload.get("sha256")
    size = payload.get("size_bytes")
    if (
        not isinstance(relative, str)
        or not relative
        or not isinstance(size, int)
        or size <= 0
    ):
        if allow_empty and relative is None and size == 0:
            return None, None, 0, None, None
        raise ValueError("stimulus payload descriptor is incomplete")
    payload_path = (stimulus_path.parent / Path(relative)).resolve()
    release_root = root.resolve()
    if release_root not in payload_path.parents or not payload_path.is_file():
        raise ValueError(f"stimulus payload escapes or is missing: {relative}")
    if payload_path.stat().st_size != size:
        raise ValueError(f"stimulus payload size mismatch: {relative}")
    try:
        parsed = parse_payload(payload_path)
    except ValueError as exc:
        raise ValueError(
            "claim timing requires a self-describing scarf-rtl-payload container with "
            "workload metadata; anonymous or unrelated payload bytes are rejected"
        ) from exc
    validate_claim_payload(parsed)
    descriptor = workload_descriptor(parsed)
    declared = payload.get("workload")
    if not isinstance(declared, dict):
        raise ValueError("stimulus payload is missing its workload descriptor")
    # The self-describing header is the binding authority. Its two provenance
    # digest fields do not affect the executable role/shape/offset contract.
    declared_binding = {
        key: value for key, value in declared.items()
        if key not in {"descriptor_sha256", "tensor_data_sha256"}
    }
    header_binding = {
        key: value for key, value in descriptor.items()
        if key not in {"descriptor_sha256", "tensor_data_sha256"}
    }
    if declared_binding != header_binding:
        raise ValueError("stimulus workload descriptor does not match payload header")
    return payload_path, digest if isinstance(digest, str) else None, size, descriptor, parsed


def run(
    stimulus_root: Path,
    source: Path,
    output: Path,
    max_cycles: int | None,
    *,
    allow_empty_payload: bool = False,
    allow_nonmonotonic: bool = False,
    diagnostic: bool = False,
) -> Path:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    stimuli = _stimuli(stimulus_root)
    output = output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scarf-verilator-") as d:
        exe = _compile(source, Path(d)); records=[]
        run_selection_sha256 = None
        diagnostic_payload_seen = False
        for item, stimulus_path in stimuli:
            model, dataset, index = item.get("model"), item.get("dataset"), item.get("sample_index")
            if not isinstance(model,str) or not isinstance(dataset,str) or not isinstance(index,int):
                raise ValueError("stimulus identity is invalid")
            payload_path, payload_digest, payload_size, descriptor, parsed_payload = _payload_for(
                item, stimulus_path, stimulus_root, allow_empty=allow_empty_payload
            )
            constant_roles = (
                constant_claim_roles(parsed_payload)
                if parsed_payload is not None else ()
            )
            # A constant FSDR feature vector intentionally drives the Full
            # route. Execute it, but do not let a degenerate diagnostic become
            # a strict timing claim simply because its bytes are well formed.
            diagnostic_payload = "pipeline_features" in constant_roles
            diagnostic_payload_seen = diagnostic_payload_seen or diagnostic_payload
            harness_payload_path = payload_path
            harness_image: bytes | None = None
            role_bindings: dict[str, tuple[int, int]] = {}
            if parsed_payload is not None:
                harness_payload_path = Path(d) / "payload.tensor-data.bin"
                harness_image = tensor_data_image(parsed_payload)
                harness_payload_path.write_bytes(harness_image)
                for tensor in descriptor["tensor_ranges"]:
                    if tensor.get("role") in {
                        "pipeline_features", "pipeline_depths", "depth_candidates",
                        "depth_probabilities",
                    }:
                        role_bindings[tensor["role"]] = (
                            tensor["offset_bytes"], tensor["size_bytes"]
                        )
                if set(role_bindings) != {
                    "pipeline_features", "pipeline_depths", "depth_candidates",
                    "depth_probabilities",
                }:
                    raise ValueError("claim payload is missing RTL role offsets")
                for tensor in descriptor["tensor_ranges"]:
                    if tensor.get("role") == "saes_routes":
                        role_bindings["saes_routes"] = (
                            tensor["offset_bytes"], tensor["size_bytes"]
                        )
            item_selection = item.get("run_selection_sha256")
            if item_selection is not None:
                if (
                    not isinstance(item_selection, str)
                    or len(item_selection) != 64
                    or any(character not in "0123456789abcdef" for character in item_selection)
                ):
                    raise ValueError("stimulus run_selection_sha256 is invalid")
                if run_selection_sha256 is None:
                    run_selection_sha256 = item_selection
                elif run_selection_sha256 != item_selection:
                    raise ValueError("timing stimuli use multiple run selections")
            variants={}; events=None; payload_reads=None; applied_registers_by_variant={}; mechanism_counters={}
            for variant in VARIANTS:
                command=[str(exe)]
                if max_cycles is not None:
                    command.extend(["--max-cycles",str(max_cycles)])
                if harness_payload_path is not None:
                    command.extend(["--payload",str(harness_payload_path)])
                if descriptor is not None:
                    command.extend([
                        "--image-h", str(descriptor["image_h"]),
                        "--image-w", str(descriptor["image_w"]),
                        "--feature-dim", str(descriptor["feature_dim"]),
                        "--num-depth", str(descriptor["num_depth_candidates"]),
                        "--num-gaussians", str(descriptor["num_gaussians"]),
                        "--payload-bytes", str(descriptor["tensor_data_size_bytes"]),
                        "--payload-tensors", str(descriptor["tensor_count"]),
                        "--feature-offset", str(role_bindings["pipeline_features"][0]),
                        "--depth-offset", str(role_bindings["pipeline_depths"][0]),
                        "--candidate-offset", str(role_bindings["depth_candidates"][0]),
                        "--probability-offset", str(role_bindings["depth_probabilities"][0]),
                        "--feature-bytes", str(role_bindings["pipeline_features"][1]),
                        "--depth-bytes", str(role_bindings["pipeline_depths"][1]),
                        "--candidate-bytes", str(role_bindings["depth_candidates"][1]),
                        "--probability-bytes", str(role_bindings["depth_probabilities"][1]),
                    ])
                    if "saes_routes" in role_bindings:
                        command.extend([
                            "--saes-route-offset", str(role_bindings["saes_routes"][0]),
                            "--saes-route-bytes", str(role_bindings["saes_routes"][1]),
                        ])
                    rtl_config = descriptor.get("rtl_config", {})
                    register_options = {
                        "tile_size": "--tile-size",
                        "cnn_layers": "--cnn-layers",
                        "transformer_layers": "--transformer-layers",
                        "norm_groups": "--norm-groups",
                        "sh_degree": "--sh-degree",
                        "has_dinov2": "--has-dinov2",
                        "dinov2_layers": "--dinov2-layers",
                        "saes_feature_var": "--saes-feature-var",
                        "saes_cross_check": "--saes-cross-check",
                        "saes_depth_std": "--saes-depth-std",
                        "fsdr_cache_size": "--fsdr-cache-size",
                        "fsdr_hamming_threshold": "--fsdr-hamming-threshold",
                        "fsdr_depth_valid_threshold": "--fsdr-depth-valid-threshold",
                    }
                    if isinstance(rtl_config, dict):
                        for field, option in register_options.items():
                            if field in rtl_config:
                                value = rtl_config[field]
                                if isinstance(value, bool):
                                    value = int(value)
                                command.extend([option, str(value)])
                if "saes" in variant: command.append("--saes")
                if "fsdr" in variant: command.append("--fsdr")
                result=subprocess.run(command,capture_output=True,text=True,check=False)
                if result.returncode: raise RuntimeError(f"RTL simulation failed for {model}/{dataset}/{index} {variant}: {result.stderr.strip()}")
                event=json.loads(result.stdout.strip().splitlines()[-1]); variants[variant]={
                    "total_cycles": int(event["total_cycles"]),
                    "stage_cycles": {
                        item["stage"]: int(item["end_cycle"]) - int(item["start_cycle"])
                        for item in event["events"]
                    },
                }
                counters = event.get("mechanism_counters")
                if not isinstance(counters, dict):
                    raise RuntimeError("RTL timing harness did not emit mechanism counters")
                mechanism_counters[variant] = counters
                rtl_config = descriptor.get("rtl_config", {}) if descriptor else {}
                applied_registers_by_variant[variant] = {
                    "0x00_num_depth_candidates": descriptor["num_depth_candidates"],
                    "0x04_feature_dim": descriptor["feature_dim"],
                    "0x08_image_h": descriptor["image_h"],
                    "0x0c_image_w": descriptor["image_w"],
                    "0x10_tile_size": rtl_config.get("tile_size", 4),
                    "0x14_cnn_layers": rtl_config.get("cnn_layers", 6),
                    "0x18_transformer_layers": rtl_config.get("transformer_layers", 6),
                    "0x1c_norm_groups": rtl_config.get("norm_groups", 8),
                    "0x20_sh_degree": rtl_config.get("sh_degree", 4),
                    "0x24_has_dinov2": int(rtl_config.get("has_dinov2", False)),
                    "0x84_dinov2_layers": rtl_config.get("dinov2_layers", 0),
                    "0x28_saes_feature_var": rtl_config.get("saes_feature_var", 0),
                    "0x2c_saes_cross_check": rtl_config.get("saes_cross_check", 0),
                    "0x30_saes_depth_std": rtl_config.get("saes_depth_std", 0),
                    "0x34_saes_enabled": int("saes" in variant),
                    "0x38_fsdr_enabled": int("fsdr" in variant),
                    "0x3c_fsdr_cache_size": rtl_config.get("fsdr_cache_size", 32),
                    "0x40_fsdr_hamming_threshold": rtl_config.get("fsdr_hamming_threshold", 3),
                    "0x44_config_valid": 1,
                    "0x48_fsdr_depth_valid_threshold": rtl_config.get("fsdr_depth_valid_threshold", 102),
                    "0x4c_payload_base": "0x90000000",
                    "0x50_payload_bytes": descriptor["tensor_data_size_bytes"],
                    "0x54_payload_tensor_count": descriptor["tensor_count"],
                    "0x58_num_gaussians": descriptor["num_gaussians"],
                    "0x60_payload_valid": 1,
                    "0x64_payload_feature_offset": role_bindings["pipeline_features"][0],
                    "0x68_payload_depth_offset": role_bindings["pipeline_depths"][0],
                    "0x6c_payload_probability_offset": role_bindings["depth_probabilities"][0],
                    "0x70_payload_candidate_offset": role_bindings["depth_candidates"][0],
                    "0x74_payload_feature_bytes": role_bindings["pipeline_features"][1],
                    "0x78_payload_depth_bytes": role_bindings["pipeline_depths"][1],
                    "0x7c_payload_candidate_bytes": role_bindings["depth_candidates"][1],
                    "0x80_payload_probability_bytes": role_bindings["depth_probabilities"][1],
                    "0x88_payload_saes_route_offset": role_bindings.get("saes_routes", (0, 0))[0],
                    "0x8c_payload_saes_route_bytes": role_bindings.get("saes_routes", (0, 0))[1],
                }
                if variant=="asic":
                    events=event["events"]
                    payload_reads={
                        "count": int(event.get("axi_read_count", 0)),
                        "addresses": event.get("axi_read_addresses", []),
                    }
            if descriptor is None:
                raise ValueError("claim timing sample has no workload descriptor")
            if not allow_nonmonotonic and not diagnostic_payload and (
                variants["asic_fsdr_saes"]["total_cycles"] >= variants["asic_fsdr"]["total_cycles"]
                or variants["asic_fsdr_saes"]["total_cycles"] >= variants["asic_saes"]["total_cycles"]
            ):
                raise RuntimeError(
                    "RTL combined mechanism variant is not strictly faster than each single-mechanism variant: "
                    + json.dumps({"cycles": variants, "mechanism_counters": mechanism_counters}, sort_keys=True)
                )
            if payload_reads is None or payload_reads["count"] <= 0:
                raise RuntimeError(
                    f"RTL consumed no payload AXI beats for {model}/{dataset}/{index}"
                )
            if len(payload_reads["addresses"]) != payload_reads["count"]:
                raise RuntimeError("RTL payload AXI read log is incomplete")
            if harness_image is None:
                raise RuntimeError("claim timing sample has no tensor-data image")
            evidence = consumption_evidence(descriptor, harness_image, payload_reads["addresses"])
            required_roles = {
                "pipeline_features", "pipeline_depths", "depth_candidates", "depth_probabilities"
            }
            if "saes_routes" in role_bindings:
                required_roles.add("saes_routes")
            evidence["role_coverage"] = {
                role: value
                for role, value in evidence["role_coverage"].items()
                if role in required_roles
            }
            if set(evidence["role_coverage"]) != required_roles:
                raise RuntimeError(
                    "RTL payload reads do not cover every critical role: "
                    + json.dumps(evidence["role_coverage"], sort_keys=True)
                )
            records.append({
                "model": model,
                "dataset": dataset,
                "sample_index": index,
                "events": events,
                "variants": variants,
                "stimulus_sha256": payload_digest,
                "stimulus_size_bytes": payload_size,
                "workload": descriptor,
                "axi_read_count": payload_reads["count"],
                "axi_read_addresses": payload_reads["addresses"],
                "role_coverage": evidence["role_coverage"],
                "axi_consumption_digest": evidence["axi_consumption_digest"],
                "mechanism_counters": mechanism_counters,
                "applied_registers": applied_registers_by_variant,
                "constant_critical_roles": list(constant_roles),
                "diagnostic_payload": diagnostic_payload,
            })
            if item_selection is not None:
                records[-1]["run_selection_sha256"] = item_selection
    event_export = {
        "schema_version": EVENT_SCHEMA,
        "clock_mhz": 1000,
        "source_rtl": str(source),
        "claim_eligible": (
            not allow_empty_payload and not allow_nonmonotonic and not diagnostic_payload_seen
            and not diagnostic
        ),
        "stimulus_consumer": "ScarfTop AXI read channel",
        "samples": records,
    }
    if run_selection_sha256 is not None:
        event_export["run_selection_sha256"] = run_selection_sha256
    output.write_text(json.dumps(event_export, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stimulus-root", type=Path, required=True)
    parser.add_argument("--source-rtl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-cycles",
        type=int,
        help="optional diagnostic limit; without it the simulator runs until io_done",
    )
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="permit metadata-only stimuli; output is explicitly non-claim",
    )
    parser.add_argument(
        "--allow-nonmonotonic",
        action="store_true",
        help="diagnostic only: export variants even when combined is not faster",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="mark route-driven representative timing as non-claim evidence",
    )
    args = parser.parse_args(argv)
    try: print(run(args.stimulus_root,args.source_rtl,args.output,args.max_cycles,allow_empty_payload=args.structural_only,allow_nonmonotonic=args.allow_nonmonotonic,diagnostic=args.diagnostic))
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError,json.JSONDecodeError,KeyError,TypeError) as exc: print(f"BLOCKED: {exc}"); return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())
