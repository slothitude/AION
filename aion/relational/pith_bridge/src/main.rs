//! pith_bridge — AION γ-residual pathway bridge over the LARQL-D1 Pith runner.
//!
//! READ-ONLY consumer of the validated R3b crate (`/mnt/seagate/r3b/runner`,
//! bit-exact through 23 stages, R3b gate green). This crate adds no engine
//! math of its own; every numeric path it exposes is the runner's own code
//! (embedding scale, run_layer composition, final norm + tied head + softcap),
//! so the Pith numerical contract `bf16-forward-v1` is honored by construction.
//!
//! Line protocol on stdin/stdout:
//!   request : one JSON object per line (see match below); "set" is followed
//!             by `4 * hidden` bytes of little-endian f32.
//!   response: one JSON line, then — for array-returning commands — the raw
//!             little-endian f32 payload.
//!
//! Commands:
//!   {"cmd":"load","checkpoint":"<dir>"}       -> meta (dims), keeps Checkpoint
//!                                                open for factor fingerprints
//!   {"cmd":"seq","tokens":[...]}              -> reset + prefill prefix + embed
//!                                                last token + PLE precompute
//!   {"cmd":"layer","layer":i}                 -> run_layer(i, h, pos, ple_i);
//!                                                payload = layer output h
//!   {"cmd":"get"}                             -> payload = current h
//!   {"cmd":"set"} (+payload)                  -> h' = payload (γ-residual add
//!                                                happens Python-side)
//!   {"cmd":"head"}                            -> payload = logits on current h
//!                                                (final_norm + tied matvec +
//!                                                softcap — the runner's exact
//!                                                head path)
//!   {"cmd":"native","tokens":[...]}           -> full prefill logits (reference)
//!   {"cmd":"fingerprint"}                     -> content hash over every factor
//!                                                matrix executed on a fixed
//!                                                probe vector + norms + embed
//!                                                rows (detects any bit change)
//!   {"cmd":"quit"}

use std::io::{BufRead, Read, Write};
use std::path::Path;

use littlebit_engine::WeightMatrix;
use runner::config::Gemma4RunnerConfig;
use runner::forward::Gemma4Runner;
use runner::loader::{load_model_weights, Checkpoint};
use runner::numeric::{bf16_mul, rms_norm};
use runner::ple;
use runner::state::{DiscardSink, ResourceBudget};

// ------------------------------------------------------------------ hashing

fn fnv_f32s(v: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &x in v {
        for b in x.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

/// Deterministic probe vector (LCG), never zero, never depends on environment.
fn probe_vec(len: usize) -> Vec<f32> {
    let mut s: u32 = 0x9e3779b9;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5
        })
        .collect()
}

// -------------------------------------------------------------------- io

fn reply(out: &mut dyn Write, line: &str) {
    out.write_all(line.as_bytes()).unwrap();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
}

fn reply_f32s(out: &mut dyn Write, line: &str, data: &[f32]) {
    reply(out, line);
    for x in data {
        out.write_all(&x.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();
}

fn read_payload(inp: &mut dyn Read, n_floats: usize) -> Vec<f32> {
    let mut raw = vec![0u8; 4 * n_floats];
    inp.read_exact(&mut raw).expect("payload read");
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

// ------------------------------------------------------------------ main

fn main() {
    let stdin = std::io::stdin();
    let mut inp = stdin.lock();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let mut ckpt: Option<Checkpoint> = None;
    let mut runner: Option<Gemma4Runner> = None;
    let mut model_ple: Option<runner::weights::PleModelWeights> = None;
    let mut cfg: Option<Gemma4RunnerConfig> = None;
    let mut h: Vec<f32> = Vec::new();
    let mut pos: usize = 0;
    let mut ple_inputs: Vec<Vec<f32>> = Vec::new();

    let mut line = String::new();
    loop {
        line.clear();
        if inp.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let v: serde_json::Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(e) => {
                reply(&mut out, &format!("{{\"ok\":false,\"err\":\"bad json: {e}\"}}"));
                continue;
            }
        };
        let cmd = v.get("cmd").and_then(|c| c.as_str()).unwrap_or("").to_string();
        match cmd.as_str() {
            "load" => {
                let dir = Path::new(v["checkpoint"].as_str().unwrap_or(""));
                let c = match Checkpoint::open(dir) {
                    Ok(c) => c,
                    Err(e) => {
                        reply(&mut out, &format!("{{\"ok\":false,\"err\":\"checkpoint: {e}\"}}"));
                        continue;
                    }
                };
                let config = Gemma4RunnerConfig::gemma4_e2b_dense_real();
                let weights = match load_model_weights(&c, &config) {
                    Ok(w) => w,
                    Err(e) => {
                        reply(&mut out, &format!("{{\"ok\":false,\"err\":\"weights: {e}\"}}"));
                        continue;
                    }
                };
                let budget = ResourceBudget::default();
                let r = match Gemma4Runner::new(
                    config.clone(),
                    weights,
                    0,
                    budget,
                    Box::new(DiscardSink),
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        reply(&mut out, &format!("{{\"ok\":false,\"err\":\"runner: {e}\"}}"));
                        continue;
                    }
                };
                model_ple = r.weights.ple.clone();
                let pd = config.ple.as_ref().map(|p| p.ple_dim).unwrap_or(0);
                let meta = format!(
                    "{{\"ok\":true,\"hidden\":{},\"layers\":{},\"vocab\":{},\
                     \"embed_scale\":{:.6},\"softcap\":{},\"ple_dim\":{},\
                     \"factor_tensors\":{}}}",
                    config.hidden_size,
                    config.num_layers,
                    config.vocab_size,
                    config.embed_scale,
                    config.final_logit_softcapping.map_or(-1.0, |c| c),
                    pd,
                    c.tensor_count,
                );
                cfg = Some(config);
                ckpt = Some(c);
                runner = Some(r);
                reply(&mut out, &meta);
            }
            "seq" => {
                let r = runner.as_mut().expect("load first");
                let config = cfg.as_ref().unwrap();
                let tokens: Vec<u32> = v["tokens"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_u64().map(|t| t as u32)).collect())
                    .unwrap_or_default();
                let n = tokens.len();
                if n == 0 {
                    reply(&mut out, "{\"ok\":false,\"err\":\"empty tokens\"}");
                    continue;
                }
                r.reset_sequence();
                pos = n - 1;
                if pos > 0 {
                    if let Err(e) = r.prefill(&tokens[..pos]) {
                        reply(&mut out, &format!("{{\"ok\":false,\"err\":\"prefill: {e}\"}}"));
                        continue;
                    }
                }
                let token = tokens[pos];
                let row = match r.weights.embed_tokens.row(token as usize) {
                    Ok(row) => row,
                    Err(e) => {
                        reply(&mut out, &format!("{{\"ok\":false,\"err\":\"embed: {e}\"}}"));
                        continue;
                    }
                };
                let h0: Vec<f32> = row.iter().map(|&v| bf16_mul(v, config.embed_scale)).collect();
                let mp = model_ple.as_ref().expect("PLE weights");
                ple_inputs = match ple::precompute(&r.config, mp, &h0, token) {
                    Ok(p) => p,
                    Err(e) => {
                        reply(&mut out, &format!("{{\"ok\":false,\"err\":\"ple: {e}\"}}"));
                        continue;
                    }
                };
                h = h0;
                reply(
                    &mut out,
                    &format!(
                        "{{\"ok\":true,\"pos\":{pos},\"hidden_dim\":{},\"ple_layers\":{}}}",
                        config.hidden_size,
                        ple_inputs.len()
                    ),
                );
            }
            "layer" => {
                let r = runner.as_mut().expect("load first");
                let i = v["layer"].as_u64().map(|x| x as usize).unwrap_or(usize::MAX);
                if i >= ple_inputs.len() {
                    reply(&mut out, &format!("{{\"ok\":false,\"err\":\"layer {i} out of range\"}}"));
                    continue;
                }
                match r.run_layer(i, &h, pos, Some(ple_inputs[i].as_slice())) {
                    Ok(hout) => {
                        h = hout;
                        reply_f32s(&mut out, &format!("{{\"ok\":true,\"layer\":{i},\"dim\":{}}}", h.len()), &h);
                    }
                    Err(e) => reply(&mut out, &format!("{{\"ok\":false,\"err\":\"layer {i}: {e}\"}}")),
                }
            }
            "get" => {
                reply_f32s(&mut out, &format!("{{\"ok\":true,\"dim\":{}}}", h.len()), &h);
            }
            "set" => {
                let newh = read_payload(&mut inp, h.len());
                h = newh;
                reply(&mut out, &format!("{{\"ok\":true,\"dim\":{}}}", h.len()));
            }
            "head" => {
                let r = runner.as_ref().expect("load first");
                let config = cfg.as_ref().unwrap();
                let h_final = rms_norm(
                    &h,
                    &r.weights.final_norm,
                    config.rms_norm_eps,
                    config.norm_weight_offset,
                );
                let mut logits = match r.weights.embed_tokens.matvec(&h_final) {
                    Ok(l) => l,
                    Err(e) => {
                        reply(&mut out, &format!("{{\"ok\":false,\"err\":\"head: {e}\"}}"));
                        continue;
                    }
                };
                if let Some(cap) = config.final_logit_softcapping {
                    for l in logits.iter_mut() {
                        *l = (l.max(-cap).min(cap) / cap).tanh() * cap;
                    }
                }
                reply_f32s(
                    &mut out,
                    &format!("{{\"ok\":true,\"vocab\":{}}}", logits.len()),
                    &logits,
                );
            }
            "native" => {
                let r = runner.as_mut().expect("load first");
                let tokens: Vec<u32> = v["tokens"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_u64().map(|t| t as u32)).collect())
                    .unwrap_or_default();
                r.reset_sequence();
                match r.prefill(&tokens) {
                    Ok(logits) => {
                        reply_f32s(
                            &mut out,
                            &format!("{{\"ok\":true,\"vocab\":{}}}", logits.len()),
                            &logits,
                        );
                    }
                    Err(e) => reply(&mut out, &format!("{{\"ok\":false,\"err\":\"native: {e}\"}}")),
                }
            }
            "fingerprint" => {
                let c = ckpt.as_ref().expect("load first");
                let config = cfg.as_ref().unwrap();
                let mut total: u64 = 0xcbf29ce484222325;
                let mut n_probes = 0usize;
                let names = [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ];
                for layer in 0..config.num_layers {
                    for name in names {
                        let wm = match c.factor_matrix(layer, name) {
                            Ok(w) => w,
                            Err(e) => {
                                reply(
                                    &mut out,
                                    &format!("{{\"ok\":false,\"err\":\"fingerprint {layer}.{name}: {e}\"}}"),
                                );
                                return;
                            }
                        };
                        let (out_f, in_f) = wm.shape();
                        let x = probe_vec(in_f);
                        let y = wm.matvec(&x).expect("probe matvec");
                        let hh = fnv_f32s(&y) ^ (out_f as u64).rotate_left(32) ^ in_f as u64;
                        total = total.wrapping_mul(0x100000001b3) ^ hh;
                        n_probes += 1;
                    }
                }
                // norms + embedding probes
                let r = runner.as_ref().expect("load first");
                total = total.wrapping_mul(0x100000001b3) ^ fnv_f32s(&r.weights.final_norm);
                for tok in [2u32, 108, 4023, 236784] {
                    let row = r.weights.embed_tokens.row(tok as usize).expect("embed probe");
                    total = total.wrapping_mul(0x100000001b3) ^ fnv_f32s(&row);
                }
                let hf = rms_norm(
                    &probe_vec(config.hidden_size),
                    &r.weights.final_norm,
                    config.rms_norm_eps,
                    config.norm_weight_offset,
                );
                let lg = r.weights.embed_tokens.matvec(&hf).expect("head probe");
                total = total.wrapping_mul(0x100000001b3) ^ fnv_f32s(&lg);
                n_probes += 1 + 4 + 1;
                reply(
                    &mut out,
                    &format!("{{\"ok\":true,\"hash\":\"{:016x}\",\"probes\":{}}}", total, n_probes),
                );
            }
            "quit" => break,
            other => {
                reply(&mut out, &format!("{{\"ok\":false,\"err\":\"unknown cmd {other}\"}}"));
            }
        }
    }
}
