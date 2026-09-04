/* Copyright (c) 2026, SpargeAttn PPU contributors. */
#pragma once

#include <cstdint>

namespace spargeattention::ppu::layout {

// Closed forms of PPU0010_16x16_Row.  dev/ppu_int8/layout_oracle.cu checks
// these against actlize's real MMA_Traits rather than trusting this header.
constexpr int accumulator_row(int lane, int value) {
  return lane / 4 + 8 * (value >> 2);
}

constexpr int accumulator_column(int lane, int value) {
  return (lane & 3) + 4 * (value & 3);
}

constexpr int accumulator_row_slot(int value) {
  return value >> 2;
}

// Physical B operand for the one-MMA C-fragment -> next-A-fragment bridge.
// The same map is independently device-anchored by fattn_ppu.cu.
constexpr uint32_t pv_bridge_b_word(int lane, int word) {
  if (word != 0 && word != 3) return 0;
  int const inner = lane & 15;
  bool const selected = (inner >> 2) == (inner & 3);
  if (!selected) return 0;
  return lane < 16 ? 0x00003c00u : 0x3c000000u;
}

}  // namespace spargeattention::ppu::layout
