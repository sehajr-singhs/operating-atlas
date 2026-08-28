---
title: Operating Atlas Explorer
emoji: 🌐
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Operating Atlas Explorer

Interactive demo for the paper: **"The shapes a machine draws: a recalibration-invariant relational fingerprint of multi-physics operation"**

## What it does

Upload any multi-channel CSV telemetry or use the built-in demos to see:

1. **All 9 relational atoms** computed live for every pair of channels
2. **The signed Levy area** — the one atom that captures lead-lag direction
3. **Time-reversal parity** — flip the record end-to-end, watch levy change sign
4. **Recalibration invariance** — warp every channel with a monotone map, atoms don't move
5. **Side-by-side comparison** with a standard correlation matrix

## The key insight

A correlation matrix is **symmetric** — swap channels and nothing changes.
The Levy area is **antisymmetric** — it flips sign. That's how it captures
lead-lag: torque heating a winding gives a positive Levy area; the reverse
gives negative. Every other relational statistic in the standard toolkit
lives in the even sector and is blind to this.

## Paper

- [Findings page](https://huggingface.co/spaces/Sejibeji/operating-atlas)
- Nature Computational Science (submitted)
