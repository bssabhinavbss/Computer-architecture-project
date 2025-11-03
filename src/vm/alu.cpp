/**
 * File Name: alu.cpp
 * Author: Vishank Singh
 * Github: https://github.com/VishankSingh
 */

#include "vm/alu.h"
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace alu {

#include <cmath> // Add this include if not already present for isnan

/**
 * @brief Converts a 32-bit float to a 16-bit bfloat16 (RTNE).
 */
uint16_t float_to_bfloat16(float f) {
    union {
        float f;
        uint32_t u;
    } un;
    un.f = f;

    // Handle NaN
    if (std::isnan(f)) {
        return 0x7FC0; // A standard qNaN bfloat16
    }

    // Get the sign bit
    uint32_t sign = (un.u >> 16) & 0x8000;

    // Add 0x7FFF (halfway point) + adjustment for "round to nearest even"
    un.u += 0x7FFF + ((un.u >> 16) & 1);

    // Return the upper 16 bits
    return (uint16_t)(sign | (un.u >> 16));
}

/**
 * @brief Converts a 16-bit bfloat16 to a 32-bit float.
 */
float bfloat16_to_float(uint16_t b) {
    union {
        uint32_t u;
        float f;
    } un;
    // Shift bfloat16 bits to the MSBs, zero the lower 16 bits
    un.u = (uint32_t)b << 16;
    return un.f;
}

static std::string decode_fclass(uint16_t res) {
  static const std::vector<std::string> labels = {
    "-infinity",   
    "-normal",      
    "-subnormal", 
    "-zero",        
    "+zero",    
    "+subnormal", 
    "+normal",    
    "+infinity",    
    "signaling NaN",
    "quiet NaN"   
  };

  std::string output;
  for (int i = 0; i < 10; i++) {
    if (res & (1 << i)) {
      if (!output.empty()) output += ", ";
      output += labels[i];
    }
  }

  return output.empty() ? "unknown" : output;
}


[[nodiscard]] std::pair<uint64_t, bool> Alu::execute(AluOp op, uint64_t a, uint64_t b) {
  switch (op) {
    case AluOp::kAdd: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      int64_t result = sa + sb;
      bool overflow = __builtin_add_overflow(sa, sb, &result);
      return {static_cast<uint64_t>(result), overflow};
    }
    case AluOp::kAddw: {
      auto sa = static_cast<int32_t>(a);
      auto sb = static_cast<int32_t>(b);
      int32_t result = sa + sb;
      bool overflow = __builtin_add_overflow(sa, sb, &result);
      return {static_cast<uint64_t>(static_cast<int32_t>(result)), overflow};
    }
    case AluOp::kSub: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      int64_t result = sa - sb;
      bool overflow = __builtin_sub_overflow(sa, sb, &result);
      return {static_cast<uint64_t>(result), overflow};
    }
    case AluOp::kSubw: {
      auto sa = static_cast<int32_t>(a);
      auto sb = static_cast<int32_t>(b);
      int32_t result = sa - sb;
      bool overflow = __builtin_sub_overflow(sa, sb, &result);
      return {static_cast<uint64_t>(static_cast<int32_t>(result)), overflow};
    }
    case AluOp::kMul: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      int64_t result = sa*sb;
      bool overflow = __builtin_mul_overflow(sa, sb, &result);
      return {static_cast<uint64_t>(result), overflow};
    }
    case AluOp::kMulh: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      // TODO: do something about this, msvc doesnt support __int128

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
      __int128 result = static_cast<__int128>(sa)*static_cast<__int128>(sb);
      auto high_result = static_cast<int64_t>(result >> 64);
#pragma GCC diagnostic pop

      return {static_cast<uint64_t>(high_result), false};
    }
    case AluOp::kMulhsu: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<uint64_t>(b);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
      __int128 result = static_cast<__int128>(sa)*static_cast<__int128>(sb);
      auto high_result = static_cast<int64_t>(result >> 64);
#pragma GCC diagnostic pop

      return {static_cast<uint64_t>(high_result), false};
    }
    case AluOp::kMulhu: {
      auto ua = static_cast<uint64_t>(a);
      auto ub = static_cast<uint64_t>(b);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
      __int128 result = static_cast<__int128>(ua)*static_cast<__int128>(ub);
      auto high_result = static_cast<int64_t>(result >> 64);
#pragma GCC diagnostic pop

      return {static_cast<uint64_t>(high_result), false};
    }
    case AluOp::kMulw: {
      auto sa = static_cast<int32_t>(a);
      auto sb = static_cast<int32_t>(b);
      int64_t result = static_cast<int64_t>(sa)*static_cast<int64_t>(sb);
      auto lower_result = static_cast<int32_t>(result);
      bool overflow = (result!=static_cast<int64_t>(static_cast<int32_t>(result)));
      return {static_cast<uint64_t>(lower_result), overflow};
    }
    case AluOp::kDiv: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      if (sb==0) {
        return {0, false};
      }
      if (sa==INT64_MIN && sb==-1) {
        return {static_cast<uint64_t>(INT64_MAX), true};
      }
      int64_t result = sa/sb;
      return {static_cast<uint64_t>(result), false};
    }
    case AluOp::kDivw: {
      auto sa = static_cast<int32_t>(a);
      auto sb = static_cast<int32_t>(b);
      if (sb==0) {
        return {0, false};
      }
      if (sa==INT32_MIN && sb==-1) {
        return {static_cast<uint64_t>(INT32_MIN), true};
      }
      int32_t result = sa/sb;
      return {static_cast<uint64_t>(result), false};
    }
    case AluOp::kDivu: {
      if (b==0) {
        return {0, false};
      }
      uint64_t result = a/b;
      return {result, false};
    }
    case AluOp::kDivuw: {
      if (b==0) {
        return {0, false};
      }
      uint64_t result = static_cast<uint32_t>(a)/static_cast<uint32_t>(b);
      return {static_cast<uint64_t>(result), false};
    }
    case AluOp::kRem: {
      if (b==0) {
        return {0, false};
      }
      int64_t result = static_cast<int64_t>(a)%static_cast<int64_t>(b);
      return {static_cast<uint64_t>(result), false};
    }
    case AluOp::kRemw: {
      if (b==0) {
        return {0, false};
      }
      int32_t result = static_cast<int32_t>(a)%static_cast<int32_t>(b);
      return {static_cast<uint64_t>(result), false};
    }
    case AluOp::kRemu: {
      if (b==0) {
        return {0, false};
      }
      uint64_t result = a%b;
      return {result, false};
    }
    case AluOp::kRemuw: {
      if (b==0) {
        return {0, false};
      }
      uint64_t result = static_cast<uint32_t>(a)%static_cast<uint32_t>(b);
      return {static_cast<uint64_t>(result), false};
    }
    case AluOp::kAnd: {
      return {static_cast<uint64_t>(a & b), false};
    }
    case AluOp::kOr: {
      return {static_cast<uint64_t>(a | b), false};
    }
    case AluOp::kXor: {
      return {static_cast<uint64_t>(a ^ b), false};
    }
    case AluOp::kAdd_simd32: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      auto sa1 = static_cast<int32_t>(sa >> 32); // upper 32
      auto sa2 = static_cast<int32_t>(sa - (sa1 << 32)); // lower 32
      auto sb1 = static_cast<int32_t>(sb >> 32); // upper 32
      auto sb2 = static_cast<int32_t>(sb - (sb1 << 32)); // lower 32
      auto sr1 = sa1 + sb1;
      auto sr2 = sa2 + sb2;
      int64_t sr = sr2 + (sr1 << 32) ;
      return{sr,false};

    }
    case AluOp::kSll: {
      uint64_t result = a << (b & 63);
      return {result, false};
    }
    case AluOp::kSllw: {
      auto sa = static_cast<uint32_t>(a);
      auto sb = static_cast<uint32_t>(b);
      uint32_t result = sa << (sb & 31);
      return {static_cast<uint64_t>(static_cast<int32_t>(result)), false};
    }
    case AluOp::kSrl: {
      uint64_t result = a >> (b & 63);
      return {result, false};
    }
    case AluOp::kSrlw: {
      auto sa = static_cast<uint32_t>(a);
      auto sb = static_cast<uint32_t>(b);
      uint32_t result = sa >> (sb & 31);
      return {static_cast<uint64_t>(static_cast<int32_t>(result)), false};
    }
    case AluOp::kSra: {
      auto sa = static_cast<int64_t>(a);
      return {static_cast<uint64_t>(sa >> (b & 63)), false};
    }
    case AluOp::kSraw: {
      auto sa = static_cast<int32_t>(a);
      return {static_cast<uint64_t>(sa >> (b & 31)), false};
    }
    case AluOp::kSlt: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      return {static_cast<uint64_t>(sa < sb), false};
    }
    case AluOp::kSltu: {
      return {static_cast<uint64_t>(a < b), false};
    }


    // And operation for SIMD 

    case AluOp::kAnd_simd4:
case AluOp::kAnd_simd2:
case AluOp::kAnd_binary: {
    return {static_cast<uint64_t>(a & b), false};
}

// Or operation for SIMD

case AluOp::kOr_simd4:
case AluOp::kOr_simd2:
case AluOp::kOr_binary: {
    return {static_cast<uint64_t>(a | b), false};
}

// XOR operation for SIMD

case AluOp::kXor_simd4:
case AluOp::kXor_simd2:
case AluOp::kXor_binary: {
    return {static_cast<uint64_t>(a ^ b), false};
}


// SIMD 4 - Add , Sub , Mul , div , rem


    case AluOp::kAdd_simd4: {
        auto sa = static_cast<int64_t>(a);
        auto sb = static_cast<int64_t>(b);

        // Extract 16 signed 4-bit lanes
        auto sa1  = static_cast<int8_t>((sa >> 0)  & 0xF);
        auto sa2  = static_cast<int8_t>((sa >> 4)  & 0xF);
        auto sa3  = static_cast<int8_t>((sa >> 8)  & 0xF);
        auto sa4  = static_cast<int8_t>((sa >> 12) & 0xF);
        auto sa5  = static_cast<int8_t>((sa >> 16) & 0xF);
        auto sa6  = static_cast<int8_t>((sa >> 20) & 0xF);
        auto sa7  = static_cast<int8_t>((sa >> 24) & 0xF);
        auto sa8  = static_cast<int8_t>((sa >> 28) & 0xF);
        auto sa9  = static_cast<int8_t>((sa >> 32) & 0xF);
        auto sa10 = static_cast<int8_t>((sa >> 36) & 0xF);
        auto sa11 = static_cast<int8_t>((sa >> 40) & 0xF);
        auto sa12 = static_cast<int8_t>((sa >> 44) & 0xF);
        auto sa13 = static_cast<int8_t>((sa >> 48) & 0xF);
        auto sa14 = static_cast<int8_t>((sa >> 52) & 0xF);
        auto sa15 = static_cast<int8_t>((sa >> 56) & 0xF);
        auto sa16 = static_cast<int8_t>((sa >> 60) & 0xF);

        auto sb1  = static_cast<int8_t>((sb >> 0)  & 0xF);
        auto sb2  = static_cast<int8_t>((sb >> 4)  & 0xF);
        auto sb3  = static_cast<int8_t>((sb >> 8)  & 0xF);
        auto sb4  = static_cast<int8_t>((sb >> 12) & 0xF);
        auto sb5  = static_cast<int8_t>((sb >> 16) & 0xF);
        auto sb6  = static_cast<int8_t>((sb >> 20) & 0xF);
        auto sb7  = static_cast<int8_t>((sb >> 24) & 0xF);
        auto sb8  = static_cast<int8_t>((sb >> 28) & 0xF);
        auto sb9  = static_cast<int8_t>((sb >> 32) & 0xF);
        auto sb10 = static_cast<int8_t>((sb >> 36) & 0xF);
        auto sb11 = static_cast<int8_t>((sb >> 40) & 0xF);
        auto sb12 = static_cast<int8_t>((sb >> 44) & 0xF);
        auto sb13 = static_cast<int8_t>((sb >> 48) & 0xF);
        auto sb14 = static_cast<int8_t>((sb >> 52) & 0xF);
        auto sb15 = static_cast<int8_t>((sb >> 56) & 0xF);
        auto sb16 = static_cast<int8_t>((sb >> 60) & 0xF);

        // Helper lambda for saturating 4-bit signed addition
        auto sat_add4 = [](int8_t x, int8_t y) -> int8_t {
            int8_t sum = x + y;
            if (sum > 7)  sum = 7;   // saturate positive
            if (sum < -8) sum = -8;  // saturate negative
            return sum & 0xF;        // keep only low 4 bits
        };

        // Compute per-lane results with saturation
        uint64_t sr = 0;
        sr |= static_cast<uint64_t>(sat_add4(sa1, sb1))  << 0;
        sr |= static_cast<uint64_t>(sat_add4(sa2, sb2))  << 4;
        sr |= static_cast<uint64_t>(sat_add4(sa3, sb3))  << 8;
        sr |= static_cast<uint64_t>(sat_add4(sa4, sb4))  << 12;
        sr |= static_cast<uint64_t>(sat_add4(sa5, sb5))  << 16;
        sr |= static_cast<uint64_t>(sat_add4(sa6, sb6))  << 20;
        sr |= static_cast<uint64_t>(sat_add4(sa7, sb7))  << 24;
        sr |= static_cast<uint64_t>(sat_add4(sa8, sb8))  << 28;
        sr |= static_cast<uint64_t>(sat_add4(sa9, sb9))  << 32;
        sr |= static_cast<uint64_t>(sat_add4(sa10, sb10)) << 36;
        sr |= static_cast<uint64_t>(sat_add4(sa11, sb11)) << 40;
        sr |= static_cast<uint64_t>(sat_add4(sa12, sb12)) << 44;
        sr |= static_cast<uint64_t>(sat_add4(sa13, sb13)) << 48;
        sr |= static_cast<uint64_t>(sat_add4(sa14, sb14)) << 52;
        sr |= static_cast<uint64_t>(sat_add4(sa15, sb15)) << 56;
        sr |= static_cast<uint64_t>(sat_add4(sa16, sb16)) << 60;

        return {sr, false};
    }


        case AluOp::kSub_simd4: {
        auto sa = static_cast<int64_t>(a);
        auto sb = static_cast<int64_t>(b);

        // Extract 16 signed 4-bit lanes
        auto sa1  = static_cast<int8_t>((sa >> 0)  & 0xF);
        auto sa2  = static_cast<int8_t>((sa >> 4)  & 0xF);
        auto sa3  = static_cast<int8_t>((sa >> 8)  & 0xF);
        auto sa4  = static_cast<int8_t>((sa >> 12) & 0xF);
        auto sa5  = static_cast<int8_t>((sa >> 16) & 0xF);
        auto sa6  = static_cast<int8_t>((sa >> 20) & 0xF);
        auto sa7  = static_cast<int8_t>((sa >> 24) & 0xF);
        auto sa8  = static_cast<int8_t>((sa >> 28) & 0xF);
        auto sa9  = static_cast<int8_t>((sa >> 32) & 0xF);
        auto sa10 = static_cast<int8_t>((sa >> 36) & 0xF);
        auto sa11 = static_cast<int8_t>((sa >> 40) & 0xF);
        auto sa12 = static_cast<int8_t>((sa >> 44) & 0xF);
        auto sa13 = static_cast<int8_t>((sa >> 48) & 0xF);
        auto sa14 = static_cast<int8_t>((sa >> 52) & 0xF);
        auto sa15 = static_cast<int8_t>((sa >> 56) & 0xF);
        auto sa16 = static_cast<int8_t>((sa >> 60) & 0xF);

        auto sb1  = static_cast<int8_t>((sb >> 0)  & 0xF);
        auto sb2  = static_cast<int8_t>((sb >> 4)  & 0xF);
        auto sb3  = static_cast<int8_t>((sb >> 8)  & 0xF);
        auto sb4  = static_cast<int8_t>((sb >> 12) & 0xF);
        auto sb5  = static_cast<int8_t>((sb >> 16) & 0xF);
        auto sb6  = static_cast<int8_t>((sb >> 20) & 0xF);
        auto sb7  = static_cast<int8_t>((sb >> 24) & 0xF);
        auto sb8  = static_cast<int8_t>((sb >> 28) & 0xF);
        auto sb9  = static_cast<int8_t>((sb >> 32) & 0xF);
        auto sb10 = static_cast<int8_t>((sb >> 36) & 0xF);
        auto sb11 = static_cast<int8_t>((sb >> 40) & 0xF);
        auto sb12 = static_cast<int8_t>((sb >> 44) & 0xF);
        auto sb13 = static_cast<int8_t>((sb >> 48) & 0xF);
        auto sb14 = static_cast<int8_t>((sb >> 52) & 0xF);
        auto sb15 = static_cast<int8_t>((sb >> 56) & 0xF);
        auto sb16 = static_cast<int8_t>((sb >> 60) & 0xF);

        // Helper lambda for saturating 4-bit signed subtraction
        auto sat_sub4 = [](int8_t x, int8_t y) -> int8_t {
            int8_t diff = x - y;
            if (diff > 7)  diff = 7;   // saturate positive
            if (diff < -8) diff = -8;  // saturate negative
            return diff & 0xF;         // keep only low 4 bits
        };

        // Compute per-lane results with saturation
        uint64_t sr = 0;
        sr |= static_cast<uint64_t>(sat_sub4(sa1, sb1))  << 0;
        sr |= static_cast<uint64_t>(sat_sub4(sa2, sb2))  << 4;
        sr |= static_cast<uint64_t>(sat_sub4(sa3, sb3))  << 8;
        sr |= static_cast<uint64_t>(sat_sub4(sa4, sb4))  << 12;
        sr |= static_cast<uint64_t>(sat_sub4(sa5, sb5))  << 16;
        sr |= static_cast<uint64_t>(sat_sub4(sa6, sb6))  << 20;
        sr |= static_cast<uint64_t>(sat_sub4(sa7, sb7))  << 24;
        sr |= static_cast<uint64_t>(sat_sub4(sa8, sb8))  << 28;
        sr |= static_cast<uint64_t>(sat_sub4(sa9, sb9))  << 32;
        sr |= static_cast<uint64_t>(sat_sub4(sa10, sb10)) << 36;
        sr |= static_cast<uint64_t>(sat_sub4(sa11, sb11)) << 40;
        sr |= static_cast<uint64_t>(sat_sub4(sa12, sb12)) << 44;
        sr |= static_cast<uint64_t>(sat_sub4(sa13, sb13)) << 48;
        sr |= static_cast<uint64_t>(sat_sub4(sa14, sb14)) << 52;
        sr |= static_cast<uint64_t>(sat_sub4(sa15, sb15)) << 56;
        sr |= static_cast<uint64_t>(sat_sub4(sa16, sb16)) << 60;

        return {sr, false};
    }


        case AluOp::kMul_simd4: {
        auto sa = static_cast<int64_t>(a);
        auto sb = static_cast<int64_t>(b);

        // Extract 16 signed 4-bit lanes
        auto sa1  = static_cast<int8_t>((sa >> 0)  & 0xF);
        auto sa2  = static_cast<int8_t>((sa >> 4)  & 0xF);
        auto sa3  = static_cast<int8_t>((sa >> 8)  & 0xF);
        auto sa4  = static_cast<int8_t>((sa >> 12) & 0xF);
        auto sa5  = static_cast<int8_t>((sa >> 16) & 0xF);
        auto sa6  = static_cast<int8_t>((sa >> 20) & 0xF);
        auto sa7  = static_cast<int8_t>((sa >> 24) & 0xF);
        auto sa8  = static_cast<int8_t>((sa >> 28) & 0xF);
        auto sa9  = static_cast<int8_t>((sa >> 32) & 0xF);
        auto sa10 = static_cast<int8_t>((sa >> 36) & 0xF);
        auto sa11 = static_cast<int8_t>((sa >> 40) & 0xF);
        auto sa12 = static_cast<int8_t>((sa >> 44) & 0xF);
        auto sa13 = static_cast<int8_t>((sa >> 48) & 0xF);
        auto sa14 = static_cast<int8_t>((sa >> 52) & 0xF);
        auto sa15 = static_cast<int8_t>((sa >> 56) & 0xF);
        auto sa16 = static_cast<int8_t>((sa >> 60) & 0xF);

        auto sb1  = static_cast<int8_t>((sb >> 0)  & 0xF);
        auto sb2  = static_cast<int8_t>((sb >> 4)  & 0xF);
        auto sb3  = static_cast<int8_t>((sb >> 8)  & 0xF);
        auto sb4  = static_cast<int8_t>((sb >> 12) & 0xF);
        auto sb5  = static_cast<int8_t>((sb >> 16) & 0xF);
        auto sb6  = static_cast<int8_t>((sb >> 20) & 0xF);
        auto sb7  = static_cast<int8_t>((sb >> 24) & 0xF);
        auto sb8  = static_cast<int8_t>((sb >> 28) & 0xF);
        auto sb9  = static_cast<int8_t>((sb >> 32) & 0xF);
        auto sb10 = static_cast<int8_t>((sb >> 36) & 0xF);
        auto sb11 = static_cast<int8_t>((sb >> 40) & 0xF);
        auto sb12 = static_cast<int8_t>((sb >> 44) & 0xF);
        auto sb13 = static_cast<int8_t>((sb >> 48) & 0xF);
        auto sb14 = static_cast<int8_t>((sb >> 52) & 0xF);
        auto sb15 = static_cast<int8_t>((sb >> 56) & 0xF);
        auto sb16 = static_cast<int8_t>((sb >> 60) & 0xF);

        // Helper lambda for saturating 4-bit signed multiply
        auto sat_mul4 = [](int8_t x, int8_t y) -> int8_t {
            // Convert to signed 4-bit first
            if (x & 0x8) x |= 0xF0;
            if (y & 0x8) y |= 0xF0;

            int16_t prod = static_cast<int16_t>(x) * static_cast<int16_t>(y);
            if (prod > 7)  prod = 7;
            if (prod < -8) prod = -8;

            return static_cast<int8_t>(prod) & 0xF;
        };

        // Compute per-lane results with saturation
        uint64_t sr = 0;
        sr |= static_cast<uint64_t>(sat_mul4(sa1, sb1))   << 0;
        sr |= static_cast<uint64_t>(sat_mul4(sa2, sb2))   << 4;
        sr |= static_cast<uint64_t>(sat_mul4(sa3, sb3))   << 8;
        sr |= static_cast<uint64_t>(sat_mul4(sa4, sb4))   << 12;
        sr |= static_cast<uint64_t>(sat_mul4(sa5, sb5))   << 16;
        sr |= static_cast<uint64_t>(sat_mul4(sa6, sb6))   << 20;
        sr |= static_cast<uint64_t>(sat_mul4(sa7, sb7))   << 24;
        sr |= static_cast<uint64_t>(sat_mul4(sa8, sb8))   << 28;
        sr |= static_cast<uint64_t>(sat_mul4(sa9, sb9))   << 32;
        sr |= static_cast<uint64_t>(sat_mul4(sa10, sb10)) << 36;
        sr |= static_cast<uint64_t>(sat_mul4(sa11, sb11)) << 40;
        sr |= static_cast<uint64_t>(sat_mul4(sa12, sb12)) << 44;
        sr |= static_cast<uint64_t>(sat_mul4(sa13, sb13)) << 48;
        sr |= static_cast<uint64_t>(sat_mul4(sa14, sb14)) << 52;
        sr |= static_cast<uint64_t>(sat_mul4(sa15, sb15)) << 56;
        sr |= static_cast<uint64_t>(sat_mul4(sa16, sb16)) << 60;

        return {sr, false};
    }

        case AluOp::kDiv_simd4: {
        auto sa = static_cast<int64_t>(a);
        auto sb = static_cast<int64_t>(b);

        // Extract 16 signed 4-bit lanes
        auto sa1  = static_cast<int8_t>((sa >> 0)  & 0xF);
        auto sa2  = static_cast<int8_t>((sa >> 4)  & 0xF);
        auto sa3  = static_cast<int8_t>((sa >> 8)  & 0xF);
        auto sa4  = static_cast<int8_t>((sa >> 12) & 0xF);
        auto sa5  = static_cast<int8_t>((sa >> 16) & 0xF);
        auto sa6  = static_cast<int8_t>((sa >> 20) & 0xF);
        auto sa7  = static_cast<int8_t>((sa >> 24) & 0xF);
        auto sa8  = static_cast<int8_t>((sa >> 28) & 0xF);
        auto sa9  = static_cast<int8_t>((sa >> 32) & 0xF);
        auto sa10 = static_cast<int8_t>((sa >> 36) & 0xF);
        auto sa11 = static_cast<int8_t>((sa >> 40) & 0xF);
        auto sa12 = static_cast<int8_t>((sa >> 44) & 0xF);
        auto sa13 = static_cast<int8_t>((sa >> 48) & 0xF);
        auto sa14 = static_cast<int8_t>((sa >> 52) & 0xF);
        auto sa15 = static_cast<int8_t>((sa >> 56) & 0xF);
        auto sa16 = static_cast<int8_t>((sa >> 60) & 0xF);

        auto sb1  = static_cast<int8_t>((sb >> 0)  & 0xF);
        auto sb2  = static_cast<int8_t>((sb >> 4)  & 0xF);
        auto sb3  = static_cast<int8_t>((sb >> 8)  & 0xF);
        auto sb4  = static_cast<int8_t>((sb >> 12) & 0xF);
        auto sb5  = static_cast<int8_t>((sb >> 16) & 0xF);
        auto sb6  = static_cast<int8_t>((sb >> 20) & 0xF);
        auto sb7  = static_cast<int8_t>((sb >> 24) & 0xF);
        auto sb8  = static_cast<int8_t>((sb >> 28) & 0xF);
        auto sb9  = static_cast<int8_t>((sb >> 32) & 0xF);
        auto sb10 = static_cast<int8_t>((sb >> 36) & 0xF);
        auto sb11 = static_cast<int8_t>((sb >> 40) & 0xF);
        auto sb12 = static_cast<int8_t>((sb >> 44) & 0xF);
        auto sb13 = static_cast<int8_t>((sb >> 48) & 0xF);
        auto sb14 = static_cast<int8_t>((sb >> 52) & 0xF);
        auto sb15 = static_cast<int8_t>((sb >> 56) & 0xF);
        auto sb16 = static_cast<int8_t>((sb >> 60) & 0xF);

        // Helper lambda for saturating 4-bit signed divide
        auto sat_div4 = [](int8_t x, int8_t y) -> int8_t {
            // Convert to signed 4-bit first
            if (x & 0x8) x |= 0xF0;
            if (y & 0x8) y |= 0xF0;

            if (y == 0)
                return static_cast<int8_t>(-1) & 0xF; // divide-by-zero -> -1 (0xF)

            int16_t quot = static_cast<int16_t>(x) / static_cast<int16_t>(y);
            if (quot > 7)  quot = 7;
            if (quot < -8) quot = -8;

            return static_cast<int8_t>(quot) & 0xF;
        };

        // Compute per-lane results with saturation
        uint64_t sr = 0;
        sr |= static_cast<uint64_t>(sat_div4(sa1, sb1))   << 0;
        sr |= static_cast<uint64_t>(sat_div4(sa2, sb2))   << 4;
        sr |= static_cast<uint64_t>(sat_div4(sa3, sb3))   << 8;
        sr |= static_cast<uint64_t>(sat_div4(sa4, sb4))   << 12;
        sr |= static_cast<uint64_t>(sat_div4(sa5, sb5))   << 16;
        sr |= static_cast<uint64_t>(sat_div4(sa6, sb6))   << 20;
        sr |= static_cast<uint64_t>(sat_div4(sa7, sb7))   << 24;
        sr |= static_cast<uint64_t>(sat_div4(sa8, sb8))   << 28;
        sr |= static_cast<uint64_t>(sat_div4(sa9, sb9))   << 32;
        sr |= static_cast<uint64_t>(sat_div4(sa10, sb10)) << 36;
        sr |= static_cast<uint64_t>(sat_div4(sa11, sb11)) << 40;
        sr |= static_cast<uint64_t>(sat_div4(sa12, sb12)) << 44;
        sr |= static_cast<uint64_t>(sat_div4(sa13, sb13)) << 48;
        sr |= static_cast<uint64_t>(sat_div4(sa14, sb14)) << 52;
        sr |= static_cast<uint64_t>(sat_div4(sa15, sb15)) << 56;
        sr |= static_cast<uint64_t>(sat_div4(sa16, sb16)) << 60;

        return {sr, false};
    }


        case AluOp::kRem_simd4: {
        auto sa = static_cast<int64_t>(a);
        auto sb = static_cast<int64_t>(b);

        // Extract 16 signed 4-bit lanes
        auto sa1  = static_cast<int8_t>((sa >> 0)  & 0xF);
        auto sa2  = static_cast<int8_t>((sa >> 4)  & 0xF);
        auto sa3  = static_cast<int8_t>((sa >> 8)  & 0xF);
        auto sa4  = static_cast<int8_t>((sa >> 12) & 0xF);
        auto sa5  = static_cast<int8_t>((sa >> 16) & 0xF);
        auto sa6  = static_cast<int8_t>((sa >> 20) & 0xF);
        auto sa7  = static_cast<int8_t>((sa >> 24) & 0xF);
        auto sa8  = static_cast<int8_t>((sa >> 28) & 0xF);
        auto sa9  = static_cast<int8_t>((sa >> 32) & 0xF);
        auto sa10 = static_cast<int8_t>((sa >> 36) & 0xF);
        auto sa11 = static_cast<int8_t>((sa >> 40) & 0xF);
        auto sa12 = static_cast<int8_t>((sa >> 44) & 0xF);
        auto sa13 = static_cast<int8_t>((sa >> 48) & 0xF);
        auto sa14 = static_cast<int8_t>((sa >> 52) & 0xF);
        auto sa15 = static_cast<int8_t>((sa >> 56) & 0xF);
        auto sa16 = static_cast<int8_t>((sa >> 60) & 0xF);

        auto sb1  = static_cast<int8_t>((sb >> 0)  & 0xF);
        auto sb2  = static_cast<int8_t>((sb >> 4)  & 0xF);
        auto sb3  = static_cast<int8_t>((sb >> 8)  & 0xF);
        auto sb4  = static_cast<int8_t>((sb >> 12) & 0xF);
        auto sb5  = static_cast<int8_t>((sb >> 16) & 0xF);
        auto sb6  = static_cast<int8_t>((sb >> 20) & 0xF);
        auto sb7  = static_cast<int8_t>((sb >> 24) & 0xF);
        auto sb8  = static_cast<int8_t>((sb >> 28) & 0xF);
        auto sb9  = static_cast<int8_t>((sb >> 32) & 0xF);
        auto sb10 = static_cast<int8_t>((sb >> 36) & 0xF);
        auto sb11 = static_cast<int8_t>((sb >> 40) & 0xF);
        auto sb12 = static_cast<int8_t>((sb >> 44) & 0xF);
        auto sb13 = static_cast<int8_t>((sb >> 48) & 0xF);
        auto sb14 = static_cast<int8_t>((sb >> 52) & 0xF);
        auto sb15 = static_cast<int8_t>((sb >> 56) & 0xF);
        auto sb16 = static_cast<int8_t>((sb >> 60) & 0xF);

        // Helper lambda for saturating 4-bit signed remainder
        auto sat_rem4 = [](int8_t x, int8_t y) -> int8_t {
            // Convert to signed 4-bit first
            if (x & 0x8) x |= 0xF0;
            if (y & 0x8) y |= 0xF0;

            if (y == 0)
                return static_cast<int8_t>(-1) & 0xF;  // div-by-zero -> -1 (0xF)

            int16_t rem = static_cast<int16_t>(x) % static_cast<int16_t>(y);
            if (rem > 7)  rem = 7;
            if (rem < -8) rem = -8;

            return static_cast<int8_t>(rem) & 0xF;
        };

        // Compute per-lane results with saturation
        uint64_t sr = 0;
        sr |= static_cast<uint64_t>(sat_rem4(sa1, sb1))   << 0;
        sr |= static_cast<uint64_t>(sat_rem4(sa2, sb2))   << 4;
        sr |= static_cast<uint64_t>(sat_rem4(sa3, sb3))   << 8;
        sr |= static_cast<uint64_t>(sat_rem4(sa4, sb4))   << 12;
        sr |= static_cast<uint64_t>(sat_rem4(sa5, sb5))   << 16;
        sr |= static_cast<uint64_t>(sat_rem4(sa6, sb6))   << 20;
        sr |= static_cast<uint64_t>(sat_rem4(sa7, sb7))   << 24;
        sr |= static_cast<uint64_t>(sat_rem4(sa8, sb8))   << 28;
        sr |= static_cast<uint64_t>(sat_rem4(sa9, sb9))   << 32;
        sr |= static_cast<uint64_t>(sat_rem4(sa10, sb10)) << 36;
        sr |= static_cast<uint64_t>(sat_rem4(sa11, sb11)) << 40;
        sr |= static_cast<uint64_t>(sat_rem4(sa12, sb12)) << 44;
        sr |= static_cast<uint64_t>(sat_rem4(sa13, sb13)) << 48;
        sr |= static_cast<uint64_t>(sat_rem4(sa14, sb14)) << 52;
        sr |= static_cast<uint64_t>(sat_rem4(sa15, sb15)) << 56;
        sr |= static_cast<uint64_t>(sat_rem4(sa16, sb16)) << 60;

        return {sr, false};
    }


    // SIMD 2 - Add , Sub , Mul , Div , Rem

case AluOp::kAdd_simd2: {
    auto sa = static_cast<int64_t>(a);
    auto sb = static_cast<int64_t>(b);

    // Extract 32 signed 2-bit lanes (lane 1 at bits 0..1, lane 2 at 2..3, ...)
    auto sa1  = static_cast<int8_t>((sa >> 0)  & 0x3);
    auto sa2  = static_cast<int8_t>((sa >> 2)  & 0x3);
    auto sa3  = static_cast<int8_t>((sa >> 4)  & 0x3);
    auto sa4  = static_cast<int8_t>((sa >> 6)  & 0x3);
    auto sa5  = static_cast<int8_t>((sa >> 8)  & 0x3);
    auto sa6  = static_cast<int8_t>((sa >> 10) & 0x3);
    auto sa7  = static_cast<int8_t>((sa >> 12) & 0x3);
    auto sa8  = static_cast<int8_t>((sa >> 14) & 0x3);
    auto sa9  = static_cast<int8_t>((sa >> 16) & 0x3);
    auto sa10 = static_cast<int8_t>((sa >> 18) & 0x3);
    auto sa11 = static_cast<int8_t>((sa >> 20) & 0x3);
    auto sa12 = static_cast<int8_t>((sa >> 22) & 0x3);
    auto sa13 = static_cast<int8_t>((sa >> 24) & 0x3);
    auto sa14 = static_cast<int8_t>((sa >> 26) & 0x3);
    auto sa15 = static_cast<int8_t>((sa >> 28) & 0x3);
    auto sa16 = static_cast<int8_t>((sa >> 30) & 0x3);
    auto sa17 = static_cast<int8_t>((sa >> 32) & 0x3);
    auto sa18 = static_cast<int8_t>((sa >> 34) & 0x3);
    auto sa19 = static_cast<int8_t>((sa >> 36) & 0x3);
    auto sa20 = static_cast<int8_t>((sa >> 38) & 0x3);
    auto sa21 = static_cast<int8_t>((sa >> 40) & 0x3);
    auto sa22 = static_cast<int8_t>((sa >> 42) & 0x3);
    auto sa23 = static_cast<int8_t>((sa >> 44) & 0x3);
    auto sa24 = static_cast<int8_t>((sa >> 46) & 0x3);
    auto sa25 = static_cast<int8_t>((sa >> 48) & 0x3);
    auto sa26 = static_cast<int8_t>((sa >> 50) & 0x3);
    auto sa27 = static_cast<int8_t>((sa >> 52) & 0x3);
    auto sa28 = static_cast<int8_t>((sa >> 54) & 0x3);
    auto sa29 = static_cast<int8_t>((sa >> 56) & 0x3);
    auto sa30 = static_cast<int8_t>((sa >> 58) & 0x3);
    auto sa31 = static_cast<int8_t>((sa >> 60) & 0x3);
    auto sa32 = static_cast<int8_t>((sa >> 62) & 0x3);

    auto sb1  = static_cast<int8_t>((sb >> 0)  & 0x3);
    auto sb2  = static_cast<int8_t>((sb >> 2)  & 0x3);
    auto sb3  = static_cast<int8_t>((sb >> 4)  & 0x3);
    auto sb4  = static_cast<int8_t>((sb >> 6)  & 0x3);
    auto sb5  = static_cast<int8_t>((sb >> 8)  & 0x3);
    auto sb6  = static_cast<int8_t>((sb >> 10) & 0x3);
    auto sb7  = static_cast<int8_t>((sb >> 12) & 0x3);
    auto sb8  = static_cast<int8_t>((sb >> 14) & 0x3);
    auto sb9  = static_cast<int8_t>((sb >> 16) & 0x3);
    auto sb10 = static_cast<int8_t>((sb >> 18) & 0x3);
    auto sb11 = static_cast<int8_t>((sb >> 20) & 0x3);
    auto sb12 = static_cast<int8_t>((sb >> 22) & 0x3);
    auto sb13 = static_cast<int8_t>((sb >> 24) & 0x3);
    auto sb14 = static_cast<int8_t>((sb >> 26) & 0x3);
    auto sb15 = static_cast<int8_t>((sb >> 28) & 0x3);
    auto sb16 = static_cast<int8_t>((sb >> 30) & 0x3);
    auto sb17 = static_cast<int8_t>((sb >> 32) & 0x3);
    auto sb18 = static_cast<int8_t>((sb >> 34) & 0x3);
    auto sb19 = static_cast<int8_t>((sb >> 36) & 0x3);
    auto sb20 = static_cast<int8_t>((sb >> 38) & 0x3);
    auto sb21 = static_cast<int8_t>((sb >> 40) & 0x3);
    auto sb22 = static_cast<int8_t>((sb >> 42) & 0x3);
    auto sb23 = static_cast<int8_t>((sb >> 44) & 0x3);
    auto sb24 = static_cast<int8_t>((sb >> 46) & 0x3);
    auto sb25 = static_cast<int8_t>((sb >> 48) & 0x3);
    auto sb26 = static_cast<int8_t>((sb >> 50) & 0x3);
    auto sb27 = static_cast<int8_t>((sb >> 52) & 0x3);
    auto sb28 = static_cast<int8_t>((sb >> 54) & 0x3);
    auto sb29 = static_cast<int8_t>((sb >> 56) & 0x3);
    auto sb30 = static_cast<int8_t>((sb >> 58) & 0x3);
    auto sb31 = static_cast<int8_t>((sb >> 60) & 0x3);
    auto sb32 = static_cast<int8_t>((sb >> 62) & 0x3);

    // Helper: sign-extend 2-bit to int8_t
    auto sign_extend2 = [](int8_t v) -> int8_t {
        // if sign bit (bit 1) set, extend
        if (v & 0x2) {
            return static_cast<int8_t>(v | 0xFC); // 0b11111100 to extend sign
        } else {
            return static_cast<int8_t>(v & 0x3);
        }
    };

    // Helper lambda for saturating 2-bit signed addition
    auto sat_add2 = [&](int8_t x_raw, int8_t y_raw) -> int8_t {
        int8_t x = sign_extend2(x_raw);
        int8_t y = sign_extend2(y_raw);
        int16_t sum = static_cast<int16_t>(x) + static_cast<int16_t>(y);
        if (sum > 1)  sum = 1;   // saturate positive (max for signed 2-bit)
        if (sum < -2) sum = -2;  // saturate negative (min for signed 2-bit)
        return static_cast<int8_t>(sum) & 0x3; // store low 2 bits
    };

    // Compute per-lane results with saturation and pack
    uint64_t sr = 0;
    sr |= static_cast<uint64_t>(sat_add2(sa1, sb1))   << 0;
    sr |= static_cast<uint64_t>(sat_add2(sa2, sb2))   << 2;
    sr |= static_cast<uint64_t>(sat_add2(sa3, sb3))   << 4;
    sr |= static_cast<uint64_t>(sat_add2(sa4, sb4))   << 6;
    sr |= static_cast<uint64_t>(sat_add2(sa5, sb5))   << 8;
    sr |= static_cast<uint64_t>(sat_add2(sa6, sb6))   << 10;
    sr |= static_cast<uint64_t>(sat_add2(sa7, sb7))   << 12;
    sr |= static_cast<uint64_t>(sat_add2(sa8, sb8))   << 14;
    sr |= static_cast<uint64_t>(sat_add2(sa9, sb9))   << 16;
    sr |= static_cast<uint64_t>(sat_add2(sa10, sb10)) << 18;
    sr |= static_cast<uint64_t>(sat_add2(sa11, sb11)) << 20;
    sr |= static_cast<uint64_t>(sat_add2(sa12, sb12)) << 22;
    sr |= static_cast<uint64_t>(sat_add2(sa13, sb13)) << 24;
    sr |= static_cast<uint64_t>(sat_add2(sa14, sb14)) << 26;
    sr |= static_cast<uint64_t>(sat_add2(sa15, sb15)) << 28;
    sr |= static_cast<uint64_t>(sat_add2(sa16, sb16)) << 30;
    sr |= static_cast<uint64_t>(sat_add2(sa17, sb17)) << 32;
    sr |= static_cast<uint64_t>(sat_add2(sa18, sb18)) << 34;
    sr |= static_cast<uint64_t>(sat_add2(sa19, sb19)) << 36;
    sr |= static_cast<uint64_t>(sat_add2(sa20, sb20)) << 38;
    sr |= static_cast<uint64_t>(sat_add2(sa21, sb21)) << 40;
    sr |= static_cast<uint64_t>(sat_add2(sa22, sb22)) << 42;
    sr |= static_cast<uint64_t>(sat_add2(sa23, sb23)) << 44;
    sr |= static_cast<uint64_t>(sat_add2(sa24, sb24)) << 46;
    sr |= static_cast<uint64_t>(sat_add2(sa25, sb25)) << 48;
    sr |= static_cast<uint64_t>(sat_add2(sa26, sb26)) << 50;
    sr |= static_cast<uint64_t>(sat_add2(sa27, sb27)) << 52;
    sr |= static_cast<uint64_t>(sat_add2(sa28, sb28)) << 54;
    sr |= static_cast<uint64_t>(sat_add2(sa29, sb29)) << 56;
    sr |= static_cast<uint64_t>(sat_add2(sa30, sb30)) << 58;
    sr |= static_cast<uint64_t>(sat_add2(sa31, sb31)) << 60;
    sr |= static_cast<uint64_t>(sat_add2(sa32, sb32)) << 62;

    return {sr, false};
}


case AluOp::kSub_simd2: {
    auto sa = static_cast<int64_t>(a);
    auto sb = static_cast<int64_t>(b);

    // Extract 32 signed 2-bit lanes (lane 1 at bits 0..1, lane 2 at 2..3, ...)
    auto sa1  = static_cast<int8_t>((sa >> 0)  & 0x3);
    auto sa2  = static_cast<int8_t>((sa >> 2)  & 0x3);
    auto sa3  = static_cast<int8_t>((sa >> 4)  & 0x3);
    auto sa4  = static_cast<int8_t>((sa >> 6)  & 0x3);
    auto sa5  = static_cast<int8_t>((sa >> 8)  & 0x3);
    auto sa6  = static_cast<int8_t>((sa >> 10) & 0x3);
    auto sa7  = static_cast<int8_t>((sa >> 12) & 0x3);
    auto sa8  = static_cast<int8_t>((sa >> 14) & 0x3);
    auto sa9  = static_cast<int8_t>((sa >> 16) & 0x3);
    auto sa10 = static_cast<int8_t>((sa >> 18) & 0x3);
    auto sa11 = static_cast<int8_t>((sa >> 20) & 0x3);
    auto sa12 = static_cast<int8_t>((sa >> 22) & 0x3);
    auto sa13 = static_cast<int8_t>((sa >> 24) & 0x3);
    auto sa14 = static_cast<int8_t>((sa >> 26) & 0x3);
    auto sa15 = static_cast<int8_t>((sa >> 28) & 0x3);
    auto sa16 = static_cast<int8_t>((sa >> 30) & 0x3);
    auto sa17 = static_cast<int8_t>((sa >> 32) & 0x3);
    auto sa18 = static_cast<int8_t>((sa >> 34) & 0x3);
    auto sa19 = static_cast<int8_t>((sa >> 36) & 0x3);
    auto sa20 = static_cast<int8_t>((sa >> 38) & 0x3);
    auto sa21 = static_cast<int8_t>((sa >> 40) & 0x3);
    auto sa22 = static_cast<int8_t>((sa >> 42) & 0x3);
    auto sa23 = static_cast<int8_t>((sa >> 44) & 0x3);
    auto sa24 = static_cast<int8_t>((sa >> 46) & 0x3);
    auto sa25 = static_cast<int8_t>((sa >> 48) & 0x3);
    auto sa26 = static_cast<int8_t>((sa >> 50) & 0x3);
    auto sa27 = static_cast<int8_t>((sa >> 52) & 0x3);
    auto sa28 = static_cast<int8_t>((sa >> 54) & 0x3);
    auto sa29 = static_cast<int8_t>((sa >> 56) & 0x3);
    auto sa30 = static_cast<int8_t>((sa >> 58) & 0x3);
    auto sa31 = static_cast<int8_t>((sa >> 60) & 0x3);
    auto sa32 = static_cast<int8_t>((sa >> 62) & 0x3);

    auto sb1  = static_cast<int8_t>((sb >> 0)  & 0x3);
    auto sb2  = static_cast<int8_t>((sb >> 2)  & 0x3);
    auto sb3  = static_cast<int8_t>((sb >> 4)  & 0x3);
    auto sb4  = static_cast<int8_t>((sb >> 6)  & 0x3);
    auto sb5  = static_cast<int8_t>((sb >> 8)  & 0x3);
    auto sb6  = static_cast<int8_t>((sb >> 10) & 0x3);
    auto sb7  = static_cast<int8_t>((sb >> 12) & 0x3);
    auto sb8  = static_cast<int8_t>((sb >> 14) & 0x3);
    auto sb9  = static_cast<int8_t>((sb >> 16) & 0x3);
    auto sb10 = static_cast<int8_t>((sb >> 18) & 0x3);
    auto sb11 = static_cast<int8_t>((sb >> 20) & 0x3);
    auto sb12 = static_cast<int8_t>((sb >> 22) & 0x3);
    auto sb13 = static_cast<int8_t>((sb >> 24) & 0x3);
    auto sb14 = static_cast<int8_t>((sb >> 26) & 0x3);
    auto sb15 = static_cast<int8_t>((sb >> 28) & 0x3);
    auto sb16 = static_cast<int8_t>((sb >> 30) & 0x3);
    auto sb17 = static_cast<int8_t>((sb >> 32) & 0x3);
    auto sb18 = static_cast<int8_t>((sb >> 34) & 0x3);
    auto sb19 = static_cast<int8_t>((sb >> 36) & 0x3);
    auto sb20 = static_cast<int8_t>((sb >> 38) & 0x3);
    auto sb21 = static_cast<int8_t>((sb >> 40) & 0x3);
    auto sb22 = static_cast<int8_t>((sb >> 42) & 0x3);
    auto sb23 = static_cast<int8_t>((sb >> 44) & 0x3);
    auto sb24 = static_cast<int8_t>((sb >> 46) & 0x3);
    auto sb25 = static_cast<int8_t>((sb >> 48) & 0x3);
    auto sb26 = static_cast<int8_t>((sb >> 50) & 0x3);
    auto sb27 = static_cast<int8_t>((sb >> 52) & 0x3);
    auto sb28 = static_cast<int8_t>((sb >> 54) & 0x3);
    auto sb29 = static_cast<int8_t>((sb >> 56) & 0x3);
    auto sb30 = static_cast<int8_t>((sb >> 58) & 0x3);
    auto sb31 = static_cast<int8_t>((sb >> 60) & 0x3);
    auto sb32 = static_cast<int8_t>((sb >> 62) & 0x3);

    // Helper: sign-extend 2-bit to int8_t
    auto sign_extend2 = [](int8_t v) -> int8_t {
        if (v & 0x2) {
            return static_cast<int8_t>(v | 0xFC); // extend sign
        } else {
            return static_cast<int8_t>(v & 0x3);
        }
    };

    // Helper lambda for saturating 2-bit signed subtraction
    auto sat_sub2 = [&](int8_t x_raw, int8_t y_raw) -> int8_t {
        int8_t x = sign_extend2(x_raw);
        int8_t y = sign_extend2(y_raw);
        int16_t diff = static_cast<int16_t>(x) - static_cast<int16_t>(y);
        if (diff > 1)  diff = 1;   // saturate positive
        if (diff < -2) diff = -2;  // saturate negative
        return static_cast<int8_t>(diff) & 0x3;
    };

    // Compute per-lane results with saturation and pack
    uint64_t sr = 0;
    sr |= static_cast<uint64_t>(sat_sub2(sa1, sb1))   << 0;
    sr |= static_cast<uint64_t>(sat_sub2(sa2, sb2))   << 2;
    sr |= static_cast<uint64_t>(sat_sub2(sa3, sb3))   << 4;
    sr |= static_cast<uint64_t>(sat_sub2(sa4, sb4))   << 6;
    sr |= static_cast<uint64_t>(sat_sub2(sa5, sb5))   << 8;
    sr |= static_cast<uint64_t>(sat_sub2(sa6, sb6))   << 10;
    sr |= static_cast<uint64_t>(sat_sub2(sa7, sb7))   << 12;
    sr |= static_cast<uint64_t>(sat_sub2(sa8, sb8))   << 14;
    sr |= static_cast<uint64_t>(sat_sub2(sa9, sb9))   << 16;
    sr |= static_cast<uint64_t>(sat_sub2(sa10, sb10)) << 18;
    sr |= static_cast<uint64_t>(sat_sub2(sa11, sb11)) << 20;
    sr |= static_cast<uint64_t>(sat_sub2(sa12, sb12)) << 22;
    sr |= static_cast<uint64_t>(sat_sub2(sa13, sb13)) << 24;
    sr |= static_cast<uint64_t>(sat_sub2(sa14, sb14)) << 26;
    sr |= static_cast<uint64_t>(sat_sub2(sa15, sb15)) << 28;
    sr |= static_cast<uint64_t>(sat_sub2(sa16, sb16)) << 30;
    sr |= static_cast<uint64_t>(sat_sub2(sa17, sb17)) << 32;
    sr |= static_cast<uint64_t>(sat_sub2(sa18, sb18)) << 34;
    sr |= static_cast<uint64_t>(sat_sub2(sa19, sb19)) << 36;
    sr |= static_cast<uint64_t>(sat_sub2(sa20, sb20)) << 38;
    sr |= static_cast<uint64_t>(sat_sub2(sa21, sb21)) << 40;
    sr |= static_cast<uint64_t>(sat_sub2(sa22, sb22)) << 42;
    sr |= static_cast<uint64_t>(sat_sub2(sa23, sb23)) << 44;
    sr |= static_cast<uint64_t>(sat_sub2(sa24, sb24)) << 46;
    sr |= static_cast<uint64_t>(sat_sub2(sa25, sb25)) << 48;
    sr |= static_cast<uint64_t>(sat_sub2(sa26, sb26)) << 50;
    sr |= static_cast<uint64_t>(sat_sub2(sa27, sb27)) << 52;
    sr |= static_cast<uint64_t>(sat_sub2(sa28, sb28)) << 54;
    sr |= static_cast<uint64_t>(sat_sub2(sa29, sb29)) << 56;
    sr |= static_cast<uint64_t>(sat_sub2(sa30, sb30)) << 58;
    sr |= static_cast<uint64_t>(sat_sub2(sa31, sb31)) << 60;
    sr |= static_cast<uint64_t>(sat_sub2(sa32, sb32)) << 62;

    return {sr, false};
}
case AluOp::kMul_simd2: {
    auto sa = static_cast<int64_t>(a);
    auto sb = static_cast<int64_t>(b);

    // Extract 32 signed 2-bit lanes
    auto sa1  = static_cast<int8_t>((sa >> 0)  & 0x3);
    auto sa2  = static_cast<int8_t>((sa >> 2)  & 0x3);
    auto sa3  = static_cast<int8_t>((sa >> 4)  & 0x3);
    auto sa4  = static_cast<int8_t>((sa >> 6)  & 0x3);
    auto sa5  = static_cast<int8_t>((sa >> 8)  & 0x3);
    auto sa6  = static_cast<int8_t>((sa >> 10) & 0x3);
    auto sa7  = static_cast<int8_t>((sa >> 12) & 0x3);
    auto sa8  = static_cast<int8_t>((sa >> 14) & 0x3);
    auto sa9  = static_cast<int8_t>((sa >> 16) & 0x3);
    auto sa10 = static_cast<int8_t>((sa >> 18) & 0x3);
    auto sa11 = static_cast<int8_t>((sa >> 20) & 0x3);
    auto sa12 = static_cast<int8_t>((sa >> 22) & 0x3);
    auto sa13 = static_cast<int8_t>((sa >> 24) & 0x3);
    auto sa14 = static_cast<int8_t>((sa >> 26) & 0x3);
    auto sa15 = static_cast<int8_t>((sa >> 28) & 0x3);
    auto sa16 = static_cast<int8_t>((sa >> 30) & 0x3);
    auto sa17 = static_cast<int8_t>((sa >> 32) & 0x3);
    auto sa18 = static_cast<int8_t>((sa >> 34) & 0x3);
    auto sa19 = static_cast<int8_t>((sa >> 36) & 0x3);
    auto sa20 = static_cast<int8_t>((sa >> 38) & 0x3);
    auto sa21 = static_cast<int8_t>((sa >> 40) & 0x3);
    auto sa22 = static_cast<int8_t>((sa >> 42) & 0x3);
    auto sa23 = static_cast<int8_t>((sa >> 44) & 0x3);
    auto sa24 = static_cast<int8_t>((sa >> 46) & 0x3);
    auto sa25 = static_cast<int8_t>((sa >> 48) & 0x3);
    auto sa26 = static_cast<int8_t>((sa >> 50) & 0x3);
    auto sa27 = static_cast<int8_t>((sa >> 52) & 0x3);
    auto sa28 = static_cast<int8_t>((sa >> 54) & 0x3);
    auto sa29 = static_cast<int8_t>((sa >> 56) & 0x3);
    auto sa30 = static_cast<int8_t>((sa >> 58) & 0x3);
    auto sa31 = static_cast<int8_t>((sa >> 60) & 0x3);
    auto sa32 = static_cast<int8_t>((sa >> 62) & 0x3);

    auto sb1  = static_cast<int8_t>((sb >> 0)  & 0x3);
    auto sb2  = static_cast<int8_t>((sb >> 2)  & 0x3);
    auto sb3  = static_cast<int8_t>((sb >> 4)  & 0x3);
    auto sb4  = static_cast<int8_t>((sb >> 6)  & 0x3);
    auto sb5  = static_cast<int8_t>((sb >> 8)  & 0x3);
    auto sb6  = static_cast<int8_t>((sb >> 10) & 0x3);
    auto sb7  = static_cast<int8_t>((sb >> 12) & 0x3);
    auto sb8  = static_cast<int8_t>((sb >> 14) & 0x3);
    auto sb9  = static_cast<int8_t>((sb >> 16) & 0x3);
    auto sb10 = static_cast<int8_t>((sb >> 18) & 0x3);
    auto sb11 = static_cast<int8_t>((sb >> 20) & 0x3);
    auto sb12 = static_cast<int8_t>((sb >> 22) & 0x3);
    auto sb13 = static_cast<int8_t>((sb >> 24) & 0x3);
    auto sb14 = static_cast<int8_t>((sb >> 26) & 0x3);
    auto sb15 = static_cast<int8_t>((sb >> 28) & 0x3);
    auto sb16 = static_cast<int8_t>((sb >> 30) & 0x3);
    auto sb17 = static_cast<int8_t>((sb >> 32) & 0x3);
    auto sb18 = static_cast<int8_t>((sb >> 34) & 0x3);
    auto sb19 = static_cast<int8_t>((sb >> 36) & 0x3);
    auto sb20 = static_cast<int8_t>((sb >> 38) & 0x3);
    auto sb21 = static_cast<int8_t>((sb >> 40) & 0x3);
    auto sb22 = static_cast<int8_t>((sb >> 42) & 0x3);
    auto sb23 = static_cast<int8_t>((sb >> 44) & 0x3);
    auto sb24 = static_cast<int8_t>((sb >> 46) & 0x3);
    auto sb25 = static_cast<int8_t>((sb >> 48) & 0x3);
    auto sb26 = static_cast<int8_t>((sb >> 50) & 0x3);
    auto sb27 = static_cast<int8_t>((sb >> 52) & 0x3);
    auto sb28 = static_cast<int8_t>((sb >> 54) & 0x3);
    auto sb29 = static_cast<int8_t>((sb >> 56) & 0x3);
    auto sb30 = static_cast<int8_t>((sb >> 58) & 0x3);
    auto sb31 = static_cast<int8_t>((sb >> 60) & 0x3);
    auto sb32 = static_cast<int8_t>((sb >> 62) & 0x3);

    // Helper: sign-extend 2-bit to int8_t
    auto sign_extend2 = [](int8_t v) -> int8_t {
        if (v & 0x2) {
            return static_cast<int8_t>(v | 0xFC); // extend sign
        } else {
            return static_cast<int8_t>(v & 0x3);
        }
    };

    // Helper lambda for saturating 2-bit signed multiplication
    auto sat_mul2 = [&](int8_t x_raw, int8_t y_raw) -> int8_t {
        int8_t x = sign_extend2(x_raw);
        int8_t y = sign_extend2(y_raw);
        int16_t prod = static_cast<int16_t>(x) * static_cast<int16_t>(y);
        if (prod > 1)  prod = 1;   // saturate positive
        if (prod < -2) prod = -2;  // saturate negative
        return static_cast<int8_t>(prod) & 0x3;
    };

    // Compute per-lane results with saturation and pack
    uint64_t sr = 0;
    sr |= static_cast<uint64_t>(sat_mul2(sa1, sb1))   << 0;
    sr |= static_cast<uint64_t>(sat_mul2(sa2, sb2))   << 2;
    sr |= static_cast<uint64_t>(sat_mul2(sa3, sb3))   << 4;
    sr |= static_cast<uint64_t>(sat_mul2(sa4, sb4))   << 6;
    sr |= static_cast<uint64_t>(sat_mul2(sa5, sb5))   << 8;
    sr |= static_cast<uint64_t>(sat_mul2(sa6, sb6))   << 10;
    sr |= static_cast<uint64_t>(sat_mul2(sa7, sb7))   << 12;
    sr |= static_cast<uint64_t>(sat_mul2(sa8, sb8))   << 14;
    sr |= static_cast<uint64_t>(sat_mul2(sa9, sb9))   << 16;
    sr |= static_cast<uint64_t>(sat_mul2(sa10, sb10)) << 18;
    sr |= static_cast<uint64_t>(sat_mul2(sa11, sb11)) << 20;
    sr |= static_cast<uint64_t>(sat_mul2(sa12, sb12)) << 22;
    sr |= static_cast<uint64_t>(sat_mul2(sa13, sb13)) << 24;
    sr |= static_cast<uint64_t>(sat_mul2(sa14, sb14)) << 26;
    sr |= static_cast<uint64_t>(sat_mul2(sa15, sb15)) << 28;
    sr |= static_cast<uint64_t>(sat_mul2(sa16, sb16)) << 30;
    sr |= static_cast<uint64_t>(sat_mul2(sa17, sb17)) << 32;
    sr |= static_cast<uint64_t>(sat_mul2(sa18, sb18)) << 34;
    sr |= static_cast<uint64_t>(sat_mul2(sa19, sb19)) << 36;
    sr |= static_cast<uint64_t>(sat_mul2(sa20, sb20)) << 38;
    sr |= static_cast<uint64_t>(sat_mul2(sa21, sb21)) << 40;
    sr |= static_cast<uint64_t>(sat_mul2(sa22, sb22)) << 42;
    sr |= static_cast<uint64_t>(sat_mul2(sa23, sb23)) << 44;
    sr |= static_cast<uint64_t>(sat_mul2(sa24, sb24)) << 46;
    sr |= static_cast<uint64_t>(sat_mul2(sa25, sb25)) << 48;
    sr |= static_cast<uint64_t>(sat_mul2(sa26, sb26)) << 50;
    sr |= static_cast<uint64_t>(sat_mul2(sa27, sb27)) << 52;
    sr |= static_cast<uint64_t>(sat_mul2(sa28, sb28)) << 54;
    sr |= static_cast<uint64_t>(sat_mul2(sa29, sb29)) << 56;
    sr |= static_cast<uint64_t>(sat_mul2(sa30, sb30)) << 58;
    sr |= static_cast<uint64_t>(sat_mul2(sa31, sb31)) << 60;
    sr |= static_cast<uint64_t>(sat_mul2(sa32, sb32)) << 62;

    return {sr, false};
}


case AluOp::kDiv_simd2: {
    auto sa = static_cast<int64_t>(a);
    auto sb = static_cast<int64_t>(b);

    // Extract 32 signed 2-bit lanes
    auto sa1  = static_cast<int8_t>((sa >> 0)  & 0x3);
    auto sa2  = static_cast<int8_t>((sa >> 2)  & 0x3);
    auto sa3  = static_cast<int8_t>((sa >> 4)  & 0x3);
    auto sa4  = static_cast<int8_t>((sa >> 6)  & 0x3);
    auto sa5  = static_cast<int8_t>((sa >> 8)  & 0x3);
    auto sa6  = static_cast<int8_t>((sa >> 10) & 0x3);
    auto sa7  = static_cast<int8_t>((sa >> 12) & 0x3);
    auto sa8  = static_cast<int8_t>((sa >> 14) & 0x3);
    auto sa9  = static_cast<int8_t>((sa >> 16) & 0x3);
    auto sa10 = static_cast<int8_t>((sa >> 18) & 0x3);
    auto sa11 = static_cast<int8_t>((sa >> 20) & 0x3);
    auto sa12 = static_cast<int8_t>((sa >> 22) & 0x3);
    auto sa13 = static_cast<int8_t>((sa >> 24) & 0x3);
    auto sa14 = static_cast<int8_t>((sa >> 26) & 0x3);
    auto sa15 = static_cast<int8_t>((sa >> 28) & 0x3);
    auto sa16 = static_cast<int8_t>((sa >> 30) & 0x3);
    auto sa17 = static_cast<int8_t>((sa >> 32) & 0x3);
    auto sa18 = static_cast<int8_t>((sa >> 34) & 0x3);
    auto sa19 = static_cast<int8_t>((sa >> 36) & 0x3);
    auto sa20 = static_cast<int8_t>((sa >> 38) & 0x3);
    auto sa21 = static_cast<int8_t>((sa >> 40) & 0x3);
    auto sa22 = static_cast<int8_t>((sa >> 42) & 0x3);
    auto sa23 = static_cast<int8_t>((sa >> 44) & 0x3);
    auto sa24 = static_cast<int8_t>((sa >> 46) & 0x3);
    auto sa25 = static_cast<int8_t>((sa >> 48) & 0x3);
    auto sa26 = static_cast<int8_t>((sa >> 50) & 0x3);
    auto sa27 = static_cast<int8_t>((sa >> 52) & 0x3);
    auto sa28 = static_cast<int8_t>((sa >> 54) & 0x3);
    auto sa29 = static_cast<int8_t>((sa >> 56) & 0x3);
    auto sa30 = static_cast<int8_t>((sa >> 58) & 0x3);
    auto sa31 = static_cast<int8_t>((sa >> 60) & 0x3);
    auto sa32 = static_cast<int8_t>((sa >> 62) & 0x3);

    auto sb1  = static_cast<int8_t>((sb >> 0)  & 0x3);
    auto sb2  = static_cast<int8_t>((sb >> 2)  & 0x3);
    auto sb3  = static_cast<int8_t>((sb >> 4)  & 0x3);
    auto sb4  = static_cast<int8_t>((sb >> 6)  & 0x3);
    auto sb5  = static_cast<int8_t>((sb >> 8)  & 0x3);
    auto sb6  = static_cast<int8_t>((sb >> 10) & 0x3);
    auto sb7  = static_cast<int8_t>((sb >> 12) & 0x3);
    auto sb8  = static_cast<int8_t>((sb >> 14) & 0x3);
    auto sb9  = static_cast<int8_t>((sb >> 16) & 0x3);
    auto sb10 = static_cast<int8_t>((sb >> 18) & 0x3);
    auto sb11 = static_cast<int8_t>((sb >> 20) & 0x3);
    auto sb12 = static_cast<int8_t>((sb >> 22) & 0x3);
    auto sb13 = static_cast<int8_t>((sb >> 24) & 0x3);
    auto sb14 = static_cast<int8_t>((sb >> 26) & 0x3);
    auto sb15 = static_cast<int8_t>((sb >> 28) & 0x3);
    auto sb16 = static_cast<int8_t>((sb >> 30) & 0x3);
    auto sb17 = static_cast<int8_t>((sb >> 32) & 0x3);
    auto sb18 = static_cast<int8_t>((sb >> 34) & 0x3);
    auto sb19 = static_cast<int8_t>((sb >> 36) & 0x3);
    auto sb20 = static_cast<int8_t>((sb >> 38) & 0x3);
    auto sb21 = static_cast<int8_t>((sb >> 40) & 0x3);
    auto sb22 = static_cast<int8_t>((sb >> 42) & 0x3);
    auto sb23 = static_cast<int8_t>((sb >> 44) & 0x3);
    auto sb24 = static_cast<int8_t>((sb >> 46) & 0x3);
    auto sb25 = static_cast<int8_t>((sb >> 48) & 0x3);
    auto sb26 = static_cast<int8_t>((sb >> 50) & 0x3);
    auto sb27 = static_cast<int8_t>((sb >> 52) & 0x3);
    auto sb28 = static_cast<int8_t>((sb >> 54) & 0x3);
    auto sb29 = static_cast<int8_t>((sb >> 56) & 0x3);
    auto sb30 = static_cast<int8_t>((sb >> 58) & 0x3);
    auto sb31 = static_cast<int8_t>((sb >> 60) & 0x3);
    auto sb32 = static_cast<int8_t>((sb >> 62) & 0x3);

    // Helper: sign-extend 2-bit to int8_t
    auto sign_extend2 = [](int8_t v) -> int8_t {
        if (v & 0x2) {
            return static_cast<int8_t>(v | 0xFC);
        } else {
            return static_cast<int8_t>(v & 0x3);
        }
    };

    // Helper lambda for saturating 2-bit signed division
    auto sat_div2 = [&](int8_t x_raw, int8_t y_raw) -> int8_t {
        int8_t x = sign_extend2(x_raw);
        int8_t y = sign_extend2(y_raw);
        int16_t div;
        if (y == 0) {
            div = 1;  // divide by zero → saturate positive
        } else {
            div = static_cast<int16_t>(x) / static_cast<int16_t>(y);
        }
        if (div > 1)  div = 1;
        if (div < -2) div = -2;
        return static_cast<int8_t>(div) & 0x3;
    };

    // Compute per-lane results with saturation and pack
    uint64_t sr = 0;
    sr |= static_cast<uint64_t>(sat_div2(sa1, sb1))   << 0;
    sr |= static_cast<uint64_t>(sat_div2(sa2, sb2))   << 2;
    sr |= static_cast<uint64_t>(sat_div2(sa3, sb3))   << 4;
    sr |= static_cast<uint64_t>(sat_div2(sa4, sb4))   << 6;
    sr |= static_cast<uint64_t>(sat_div2(sa5, sb5))   << 8;
    sr |= static_cast<uint64_t>(sat_div2(sa6, sb6))   << 10;
    sr |= static_cast<uint64_t>(sat_div2(sa7, sb7))   << 12;
    sr |= static_cast<uint64_t>(sat_div2(sa8, sb8))   << 14;
    sr |= static_cast<uint64_t>(sat_div2(sa9, sb9))   << 16;
    sr |= static_cast<uint64_t>(sat_div2(sa10, sb10)) << 18;
    sr |= static_cast<uint64_t>(sat_div2(sa11, sb11)) << 20;
    sr |= static_cast<uint64_t>(sat_div2(sa12, sb12)) << 22;
    sr |= static_cast<uint64_t>(sat_div2(sa13, sb13)) << 24;
    sr |= static_cast<uint64_t>(sat_div2(sa14, sb14)) << 26;
    sr |= static_cast<uint64_t>(sat_div2(sa15, sb15)) << 28;
    sr |= static_cast<uint64_t>(sat_div2(sa16, sb16)) << 30;
    sr |= static_cast<uint64_t>(sat_div2(sa17, sb17)) << 32;
    sr |= static_cast<uint64_t>(sat_div2(sa18, sb18)) << 34;
    sr |= static_cast<uint64_t>(sat_div2(sa19, sb19)) << 36;
    sr |= static_cast<uint64_t>(sat_div2(sa20, sb20)) << 38;
    sr |= static_cast<uint64_t>(sat_div2(sa21, sb21)) << 40;
    sr |= static_cast<uint64_t>(sat_div2(sa22, sb22)) << 42;
    sr |= static_cast<uint64_t>(sat_div2(sa23, sb23)) << 44;
    sr |= static_cast<uint64_t>(sat_div2(sa24, sb24)) << 46;
    sr |= static_cast<uint64_t>(sat_div2(sa25, sb25)) << 48;
    sr |= static_cast<uint64_t>(sat_div2(sa26, sb26)) << 50;
    sr |= static_cast<uint64_t>(sat_div2(sa27, sb27)) << 52;
    sr |= static_cast<uint64_t>(sat_div2(sa28, sb28)) << 54;
    sr |= static_cast<uint64_t>(sat_div2(sa29, sb29)) << 56;
    sr |= static_cast<uint64_t>(sat_div2(sa30, sb30)) << 58;
    sr |= static_cast<uint64_t>(sat_div2(sa31, sb31)) << 60;
    sr |= static_cast<uint64_t>(sat_div2(sa32, sb32)) << 62;

    return {sr, false};
}
case AluOp::kRem_simd2: {
    auto sa = static_cast<int64_t>(a);
    auto sb = static_cast<int64_t>(b);

    // Extract 32 signed 2-bit lanes
    auto sa1  = static_cast<int8_t>((sa >> 0)  & 0x3);
    auto sa2  = static_cast<int8_t>((sa >> 2)  & 0x3);
    auto sa3  = static_cast<int8_t>((sa >> 4)  & 0x3);
    auto sa4  = static_cast<int8_t>((sa >> 6)  & 0x3);
    auto sa5  = static_cast<int8_t>((sa >> 8)  & 0x3);
    auto sa6  = static_cast<int8_t>((sa >> 10) & 0x3);
    auto sa7  = static_cast<int8_t>((sa >> 12) & 0x3);
    auto sa8  = static_cast<int8_t>((sa >> 14) & 0x3);
    auto sa9  = static_cast<int8_t>((sa >> 16) & 0x3);
    auto sa10 = static_cast<int8_t>((sa >> 18) & 0x3);
    auto sa11 = static_cast<int8_t>((sa >> 20) & 0x3);
    auto sa12 = static_cast<int8_t>((sa >> 22) & 0x3);
    auto sa13 = static_cast<int8_t>((sa >> 24) & 0x3);
    auto sa14 = static_cast<int8_t>((sa >> 26) & 0x3);
    auto sa15 = static_cast<int8_t>((sa >> 28) & 0x3);
    auto sa16 = static_cast<int8_t>((sa >> 30) & 0x3);
    auto sa17 = static_cast<int8_t>((sa >> 32) & 0x3);
    auto sa18 = static_cast<int8_t>((sa >> 34) & 0x3);
    auto sa19 = static_cast<int8_t>((sa >> 36) & 0x3);
    auto sa20 = static_cast<int8_t>((sa >> 38) & 0x3);
    auto sa21 = static_cast<int8_t>((sa >> 40) & 0x3);
    auto sa22 = static_cast<int8_t>((sa >> 42) & 0x3);
    auto sa23 = static_cast<int8_t>((sa >> 44) & 0x3);
    auto sa24 = static_cast<int8_t>((sa >> 46) & 0x3);
    auto sa25 = static_cast<int8_t>((sa >> 48) & 0x3);
    auto sa26 = static_cast<int8_t>((sa >> 50) & 0x3);
    auto sa27 = static_cast<int8_t>((sa >> 52) & 0x3);
    auto sa28 = static_cast<int8_t>((sa >> 54) & 0x3);
    auto sa29 = static_cast<int8_t>((sa >> 56) & 0x3);
    auto sa30 = static_cast<int8_t>((sa >> 58) & 0x3);
    auto sa31 = static_cast<int8_t>((sa >> 60) & 0x3);
    auto sa32 = static_cast<int8_t>((sa >> 62) & 0x3);

    auto sb1  = static_cast<int8_t>((sb >> 0)  & 0x3);
    auto sb2  = static_cast<int8_t>((sb >> 2)  & 0x3);
    auto sb3  = static_cast<int8_t>((sb >> 4)  & 0x3);
    auto sb4  = static_cast<int8_t>((sb >> 6)  & 0x3);
    auto sb5  = static_cast<int8_t>((sb >> 8)  & 0x3);
    auto sb6  = static_cast<int8_t>((sb >> 10) & 0x3);
    auto sb7  = static_cast<int8_t>((sb >> 12) & 0x3);
    auto sb8  = static_cast<int8_t>((sb >> 14) & 0x3);
    auto sb9  = static_cast<int8_t>((sb >> 16) & 0x3);
    auto sb10 = static_cast<int8_t>((sb >> 18) & 0x3);
    auto sb11 = static_cast<int8_t>((sb >> 20) & 0x3);
    auto sb12 = static_cast<int8_t>((sb >> 22) & 0x3);
    auto sb13 = static_cast<int8_t>((sb >> 24) & 0x3);
    auto sb14 = static_cast<int8_t>((sb >> 26) & 0x3);
    auto sb15 = static_cast<int8_t>((sb >> 28) & 0x3);
    auto sb16 = static_cast<int8_t>((sb >> 30) & 0x3);
    auto sb17 = static_cast<int8_t>((sb >> 32) & 0x3);
    auto sb18 = static_cast<int8_t>((sb >> 34) & 0x3);
    auto sb19 = static_cast<int8_t>((sb >> 36) & 0x3);
    auto sb20 = static_cast<int8_t>((sb >> 38) & 0x3);
    auto sb21 = static_cast<int8_t>((sb >> 40) & 0x3);
    auto sb22 = static_cast<int8_t>((sb >> 42) & 0x3);
    auto sb23 = static_cast<int8_t>((sb >> 44) & 0x3);
    auto sb24 = static_cast<int8_t>((sb >> 46) & 0x3);
    auto sb25 = static_cast<int8_t>((sb >> 48) & 0x3);
    auto sb26 = static_cast<int8_t>((sb >> 50) & 0x3);
    auto sb27 = static_cast<int8_t>((sb >> 52) & 0x3);
    auto sb28 = static_cast<int8_t>((sb >> 54) & 0x3);
    auto sb29 = static_cast<int8_t>((sb >> 56) & 0x3);
    auto sb30 = static_cast<int8_t>((sb >> 58) & 0x3);
    auto sb31 = static_cast<int8_t>((sb >> 60) & 0x3);
    auto sb32 = static_cast<int8_t>((sb >> 62) & 0x3);

    // Helper: sign-extend 2-bit to int8_t
    auto sign_extend2 = [](int8_t v) -> int8_t {
        if (v & 0x2) return static_cast<int8_t>(v | 0xFC);
        else return static_cast<int8_t>(v & 0x3);
    };

    // Helper lambda for saturating 2-bit signed remainder
    auto sat_rem2 = [&](int8_t x_raw, int8_t y_raw) -> int8_t {
        int8_t x = sign_extend2(x_raw);
        int8_t y = sign_extend2(y_raw);
        int16_t rem;
        if (y == 0) {
            rem = 1;  // division by zero → saturate positive
        } else {
            rem = static_cast<int16_t>(x) % static_cast<int16_t>(y);
        }
        if (rem > 1)  rem = 1;
        if (rem < -2) rem = -2;
        return static_cast<int8_t>(rem) & 0x3;
    };

    // Compute per-lane results with saturation and pack
    uint64_t sr = 0;
    sr |= static_cast<uint64_t>(sat_rem2(sa1, sb1))   << 0;
    sr |= static_cast<uint64_t>(sat_rem2(sa2, sb2))   << 2;
    sr |= static_cast<uint64_t>(sat_rem2(sa3, sb3))   << 4;
    sr |= static_cast<uint64_t>(sat_rem2(sa4, sb4))   << 6;
    sr |= static_cast<uint64_t>(sat_rem2(sa5, sb5))   << 8;
    sr |= static_cast<uint64_t>(sat_rem2(sa6, sb6))   << 10;
    sr |= static_cast<uint64_t>(sat_rem2(sa7, sb7))   << 12;
    sr |= static_cast<uint64_t>(sat_rem2(sa8, sb8))   << 14;
    sr |= static_cast<uint64_t>(sat_rem2(sa9, sb9))   << 16;
    sr |= static_cast<uint64_t>(sat_rem2(sa10, sb10)) << 18;
    sr |= static_cast<uint64_t>(sat_rem2(sa11, sb11)) << 20;
    sr |= static_cast<uint64_t>(sat_rem2(sa12, sb12)) << 22;
    sr |= static_cast<uint64_t>(sat_rem2(sa13, sb13)) << 24;
    sr |= static_cast<uint64_t>(sat_rem2(sa14, sb14)) << 26;
    sr |= static_cast<uint64_t>(sat_rem2(sa15, sb15)) << 28;
    sr |= static_cast<uint64_t>(sat_rem2(sa16, sb16)) << 30;
    sr |= static_cast<uint64_t>(sat_rem2(sa17, sb17)) << 32;
    sr |= static_cast<uint64_t>(sat_rem2(sa18, sb18)) << 34;
    sr |= static_cast<uint64_t>(sat_rem2(sa19, sb19)) << 36;
    sr |= static_cast<uint64_t>(sat_rem2(sa20, sb20)) << 38;
    sr |= static_cast<uint64_t>(sat_rem2(sa21, sb21)) << 40;
    sr |= static_cast<uint64_t>(sat_rem2(sa22, sb22)) << 42;
    sr |= static_cast<uint64_t>(sat_rem2(sa23, sb23)) << 44;
    sr |= static_cast<uint64_t>(sat_rem2(sa24, sb24)) << 46;
    sr |= static_cast<uint64_t>(sat_rem2(sa25, sb25)) << 48;
    sr |= static_cast<uint64_t>(sat_rem2(sa26, sb26)) << 50;
    sr |= static_cast<uint64_t>(sat_rem2(sa27, sb27)) << 52;
    sr |= static_cast<uint64_t>(sat_rem2(sa28, sb28)) << 54;
    sr |= static_cast<uint64_t>(sat_rem2(sa29, sb29)) << 56;
    sr |= static_cast<uint64_t>(sat_rem2(sa30, sb30)) << 58;
    sr |= static_cast<uint64_t>(sat_rem2(sa31, sb31)) << 60;
    sr |= static_cast<uint64_t>(sat_rem2(sa32, sb32)) << 62;

    return {sr, false};
}




    default: return {0, false};
  }
}

[[nodiscard]] std::pair<uint64_t, uint8_t> Alu::fpexecute(AluOp op,
                                                          uint64_t ina,
                                                          uint64_t inb,
                                                          uint64_t inc,
                                                          uint8_t rm) {
  float a, b, c;
  std::memcpy(&a, &ina, sizeof(float));
  std::memcpy(&b, &inb, sizeof(float));
  std::memcpy(&c, &inc, sizeof(float));
  float result = 0.0;

  uint8_t fcsr = 0;

  int original_rm = std::fegetround();

  switch (rm) {
    case 0b000: std::fesetround(FE_TONEAREST);
      break;  // RNE
    case 0b001: std::fesetround(FE_TOWARDZERO);
      break; // RTZ
    case 0b010: std::fesetround(FE_DOWNWARD);
      break;   // RDN
    case 0b011: std::fesetround(FE_UPWARD);
      break;     // RUP
      // 0b100 RMM, unsupported
    default: break;
  }

  std::feclearexcept(FE_ALL_EXCEPT);

  switch (op) {
    case AluOp::kAdd: {
      auto sa = static_cast<int64_t>(ina);
      auto sb = static_cast<int64_t>(inb);
      int64_t res = sa + sb;
      // bool overflow = __builtin_add_overflow(sa, sb, &res); // TODO: check this
      return {static_cast<uint64_t>(res), 0};
    }
    case AluOp::kFmadd_s: {
      result = std::fma(a, b, c);
      break;
    }
    case AluOp::kFmsub_s: {
      result = std::fma(a, b, -c);
      break;
    }
    case AluOp::kFnmadd_s: {
      result = std::fma(-a, b, -c);
      break;
    }
    case AluOp::kFnmsub_s: {
      result = std::fma(-a, b, c);
      break;
    }
    case AluOp::FADD_S: {
      result = a + b;
      break;
    }
    case AluOp::FSUB_S: {
      result = a - b;
      break;
    }
    case AluOp::FMUL_S: {
      result = a * b;
      break;
    }
    case AluOp::FDIV_S: {
      if (b == 0.0f) {
        result = std::numeric_limits<float>::quiet_NaN();
        fcsr |= FCSR_DIV_BY_ZERO;
      } else {
        result = a / b;
      }
      break;
    }
    case AluOp::FSQRT_S: {
      if (a < 0.0f) {
        result = std::numeric_limits<float>::quiet_NaN();
        fcsr |= FCSR_INVALID_OP;
      } else {
        result = std::sqrt(a);
      }
      break;
    }
    case AluOp::FCVT_W_S: {
      if (!std::isfinite(a) || a > static_cast<float>(INT32_MAX) || a < static_cast<float>(INT32_MIN)) {
        fcsr |= FCSR_INVALID_OP;
        fesetround(original_rm);
        auto res = static_cast<int64_t>(static_cast<int32_t>(a > 0 ? INT32_MAX : INT32_MIN));
        return {static_cast<uint64_t>(res), fcsr};
      } else {
        auto ires = static_cast<int32_t>(std::nearbyint(a));
        auto res = static_cast<int64_t>(ires); // sign-extend
        fesetround(original_rm);
        return {static_cast<uint64_t>(res), fcsr};
      }
      break;
    }
    case AluOp::FCVT_WU_S: {
      if (!std::isfinite(a) || a > static_cast<float>(UINT32_MAX) || a < 0.0f) {
        fcsr |= FCSR_INVALID_OP;
        fesetround(original_rm);
        uint32_t saturate = (a < 0.0f) ? 0 : UINT32_MAX;
        auto res = static_cast<int64_t>(static_cast<int32_t>(saturate)); // sign-extend
        return {static_cast<uint64_t>(res), fcsr};
      } else {
        auto ires = static_cast<uint32_t>(std::nearbyint(a));
        auto res = static_cast<int64_t>(static_cast<int32_t>(ires)); // sign-extend
        fesetround(original_rm);
        return {static_cast<uint64_t>(res), fcsr};
      }
      break;
    }
    case AluOp::FCVT_L_S: {
      if (!std::isfinite(a) || a > static_cast<float>(INT64_MAX) || a < static_cast<float>(INT64_MIN)) {
        fcsr |= FCSR_INVALID_OP;
        fesetround(original_rm);
        int64_t saturate = (a < 0.0f) ? INT64_MIN : INT64_MAX;
        return {static_cast<uint64_t>(saturate), fcsr};
      } else {
        auto ires = static_cast<int64_t>(std::nearbyint(a));
        fesetround(original_rm);
        return {static_cast<uint64_t>(ires), fcsr};
      }
      break;
    }
    case AluOp::FCVT_LU_S: {
      if (!std::isfinite(a) || a > static_cast<float>(UINT64_MAX) || a < 0.0f) {
        fcsr |= FCSR_INVALID_OP;
        fesetround(original_rm);
        uint64_t saturate = (a < 0.0f) ? 0 : UINT64_MAX;
        return {saturate, fcsr};
      } else {
        auto ires = static_cast<uint64_t>(std::nearbyint(a));
        fesetround(original_rm);
        return {ires, fcsr};
      }
      break;
    }
    case AluOp::FCVT_S_W: {
      auto ia = static_cast<int32_t>(ina);
      result = static_cast<float>(ia);
      break;

    }
    case AluOp::FCVT_S_WU: {
      auto ua = static_cast<uint32_t>(ina);
      result = static_cast<float>(ua);
      break;
    }
    case AluOp::FCVT_S_L: {
      auto la = static_cast<int64_t>(ina);
      result = static_cast<float>(la);
      break;

    }
    case AluOp::FCVT_S_LU: {
      auto ula = static_cast<uint64_t>(ina);
      result = static_cast<float>(ula);
      break;
    }
    case AluOp::FSGNJ_S: {
      auto a_bits = static_cast<uint32_t>(ina);
      auto b_bits = static_cast<uint32_t>(inb);
      uint32_t temp = (a_bits & 0x7FFFFFFF) | (b_bits & 0x80000000);
      std::memcpy(&result, &temp, sizeof(float));
      break;
    }
    case AluOp::FSGNJN_S: {
      auto a_bits = static_cast<uint32_t>(ina);
      auto b_bits = static_cast<uint32_t>(inb);
      uint32_t temp = (a_bits & 0x7FFFFFFF) | (~b_bits & 0x80000000);
      std::memcpy(&result, &temp, sizeof(float));
      break;
    }
    case AluOp::FSGNJX_S: {
      auto a_bits = static_cast<uint32_t>(ina);
      auto b_bits = static_cast<uint32_t>(inb);
      uint32_t temp = (a_bits & 0x7FFFFFFF) | ((a_bits ^ b_bits) & 0x80000000);
      std::memcpy(&result, &temp, sizeof(float));
      break;
    }
    case AluOp::FMIN_S: {
      if (std::isnan(a) && !std::isnan(b)) {
        result = b;
      } else if (!std::isnan(a) && std::isnan(b)) {
        result = a;
      } else if (std::signbit(a)!=std::signbit(b) && a==b) {
        result = -0.0f; // Both zero but with different signs — return -0.0
      } else {
        result = std::fmin(a, b);
      }
      break;
    }
    case AluOp::FMAX_S: {
      if (std::isnan(a) && !std::isnan(b)) {
        result = b;
      } else if (!std::isnan(a) && std::isnan(b)) {
        result = a;
      } else if (std::signbit(a)!=std::signbit(b) && a==b) {
        result = 0.0f; // Both zero but with different signs — return +0.0
      } else {
        result = std::fmax(a, b);
      }
      break;
    }
    case AluOp::FEQ_S: {
      if (std::isnan(a) || std::isnan(b)) {
        result = 0.0f;
      } else {
        result = (a==b) ? 1.0f : 0.0f;
        if (result == 1.0f) {
          return {0b1,fcsr};
        }
      }
      break;
    }
    case AluOp::FLT_S: {
      if (std::isnan(a) || std::isnan(b)) {
        result = 0.0f;
      } else {
        result = (a < b) ? 1.0f : 0.0f;
        if (result == 1.0f) {
          return {0b1,fcsr};
        }
      }
      break;
    }
    case AluOp::FLE_S: {
      if (std::isnan(a) || std::isnan(b)) {
        result = 0.0f;
      } else {
        result = (a <= b) ? 1.0f : 0.0f;
        if (result == 1.0f) {
          return {0b1, fcsr};
        }
      }
      break;
    }
    case AluOp::FCLASS_S: {
      auto a_bits = static_cast<uint32_t>(ina);
      float af;
      std::memcpy(&af, &a_bits, sizeof(float));
      uint16_t res = 0;

      if (std::signbit(af) && std::isinf(af)) res |= 1 << 0; // -inf
      else if (std::signbit(af) && std::fpclassify(af)==FP_NORMAL) res |= 1 << 1; // -normal
      else if (std::signbit(af) && std::fpclassify(af)==FP_SUBNORMAL) res |= 1 << 2; // -subnormal
      else if (std::signbit(af) && af==0.0f) res |= 1 << 3; // -zero
      else if (!std::signbit(af) && af==0.0f) res |= 1 << 4; // +zero
      else if (!std::signbit(af) && std::fpclassify(af)==FP_SUBNORMAL) res |= 1 << 5; // +subnormal
      else if (!std::signbit(af) && std::fpclassify(af)==FP_NORMAL) res |= 1 << 6; // +normal
      else if (!std::signbit(af) && std::isinf(af)) res |= 1 << 7; // +inf
      else if (std::isnan(af) && (a_bits & 0x00400000)==0) res |= 1 << 8; // signaling NaN
      else if (std::isnan(af)) res |= 1 << 9; // quiet NaN

      std::fesetround(original_rm);
      // std::cout << "Class: " << decode_fclass(res) << "\n";


      return {res, fcsr};
    }
    case AluOp::FMV_X_W: {
      int32_t float_bits;
      std::memcpy(&float_bits, &ina, sizeof(float));
      auto sign_extended = static_cast<int64_t>(float_bits);
      return {static_cast<uint64_t>(sign_extended), fcsr};
    }
    case AluOp::FMV_W_X: {
      auto int_bits = static_cast<uint32_t>(ina & 0xFFFFFFFF);
      std::memcpy(&result, &int_bits, sizeof(float));
      break;
    }

    case AluOp::FADD_BF16: {
        // Unpack 4x BF16 from ina (fs1) and inb (fs2)
        uint16_t fs1_vals[4];
        uint16_t fs2_vals[4];
        for (int i = 0; i < 4; ++i) {
            fs1_vals[i] = (uint16_t)(ina >> (i * 16));
            fs2_vals[i] = (uint16_t)(inb >> (i * 16));
        }

        float results_fp32[4];
        // Convert, Compute, Convert back
        for (int i = 0; i < 4; ++i) {
            float f1 = bfloat16_to_float(fs1_vals[i]);
            float f2 = bfloat16_to_float(fs2_vals[i]);
            results_fp32[i] = f1 + f2;
        }

        // Pack results back into a uint64_t
        uint64_t result_64 = 0;
        for (int i = 0; i < 4; ++i) {
            result_64 |= (uint64_t)float_to_bfloat16(results_fp32[i]) << (i * 16);
        }
        std::fesetround(original_rm); // Restore rounding mode
        return {result_64, fcsr}; // Return packed result, fcsr might need adjustment later
    } // No break needed after return

    case AluOp::FSUB_BF16: {
        uint16_t fs1_vals[4];
        uint16_t fs2_vals[4];
        for (int i = 0; i < 4; ++i) {
            fs1_vals[i] = (uint16_t)(ina >> (i * 16));
            fs2_vals[i] = (uint16_t)(inb >> (i * 16));
        }
        float results_fp32[4];
        for (int i = 0; i < 4; ++i) {
            float f1 = bfloat16_to_float(fs1_vals[i]);
            float f2 = bfloat16_to_float(fs2_vals[i]);
            results_fp32[i] = f1 - f2;
        }
        uint64_t result_64 = 0;
        for (int i = 0; i < 4; ++i) {
            result_64 |= (uint64_t)float_to_bfloat16(results_fp32[i]) << (i * 16);
        }
        std::fesetround(original_rm);
        return {result_64, fcsr};
    }

    case AluOp::FMUL_BF16: {
        uint16_t fs1_vals[4];
        uint16_t fs2_vals[4];
        for (int i = 0; i < 4; ++i) {
            fs1_vals[i] = (uint16_t)(ina >> (i * 16));
            fs2_vals[i] = (uint16_t)(inb >> (i * 16));
        }
        float results_fp32[4];
        for (int i = 0; i < 4; ++i) {
            float f1 = bfloat16_to_float(fs1_vals[i]);
            float f2 = bfloat16_to_float(fs2_vals[i]);
            results_fp32[i] = f1 * f2;
        }
        uint64_t result_64 = 0;
        for (int i = 0; i < 4; ++i) {
            result_64 |= (uint64_t)float_to_bfloat16(results_fp32[i]) << (i * 16);
        }
        std::fesetround(original_rm);
        return {result_64, fcsr};
    }

    case AluOp::FMAX_BF16: {
        uint16_t fs1_vals[4];
        uint16_t fs2_vals[4];
        for (int i = 0; i < 4; ++i) {
            fs1_vals[i] = (uint16_t)(ina >> (i * 16));
            fs2_vals[i] = (uint16_t)(inb >> (i * 16));
        }
        float results_fp32[4];
        for (int i = 0; i < 4; ++i) {
            float f1 = bfloat16_to_float(fs1_vals[i]);
            float f2 = bfloat16_to_float(fs2_vals[i]);
            // Handle NaNs according to standard fmax behavior if necessary
            results_fp32[i] = (f1 > f2) ? f1 : f2; // Simplified max
        }
        uint64_t result_64 = 0;
        for (int i = 0; i < 4; ++i) {
            result_64 |= (uint64_t)float_to_bfloat16(results_fp32[i]) << (i * 16);
        }
        std::fesetround(original_rm);
        return {result_64, fcsr};
    }

    case AluOp::FMADD_BF16: {
        // Unpack fs1 (ina), fs2 (inb), fs3 (inc)
        uint16_t fs1_vals[4];
        uint16_t fs2_vals[4];
        uint16_t fs3_vals[4];
        for (int i = 0; i < 4; ++i) {
            fs1_vals[i] = (uint16_t)(ina >> (i * 16));
            fs2_vals[i] = (uint16_t)(inb >> (i * 16));
            fs3_vals[i] = (uint16_t)(inc >> (i * 16)); // Use inc for fs3
        }

        float results_fp32[4];
        for (int i = 0; i < 4; ++i) {
            float f1 = bfloat16_to_float(fs1_vals[i]);
            float f2 = bfloat16_to_float(fs2_vals[i]);
            float f3 = bfloat16_to_float(fs3_vals[i]);
            results_fp32[i] = std::fma(f1, f2, f3); // Use fma for fused multiply-add
        }

        uint64_t result_64 = 0;
        for (int i = 0; i < 4; ++i) {
            result_64 |= (uint64_t)float_to_bfloat16(results_fp32[i]) << (i * 16);
        }
        std::fesetround(original_rm);
        return {result_64, fcsr};
    }
    default: break;
  }

  int raised = std::fetestexcept(FE_ALL_EXCEPT);
  if (raised & FE_INVALID) fcsr |= FCSR_INVALID_OP;
  if (raised & FE_DIVBYZERO) fcsr |= FCSR_DIV_BY_ZERO;
  if (raised & FE_OVERFLOW) fcsr |= FCSR_OVERFLOW;
  if (raised & FE_UNDERFLOW) fcsr |= FCSR_UNDERFLOW;
  if (raised & FE_INEXACT) fcsr |= FCSR_INEXACT;

  std::fesetround(original_rm);

  uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result, sizeof(result));
  return {static_cast<uint64_t>(result_bits), fcsr};
}

[[nodiscard]] std::pair<uint64_t, bool> Alu::dfpexecute(AluOp op,
                                                        uint64_t ina,
                                                        uint64_t inb,
                                                        uint64_t inc,
                                                        uint8_t rm) {
  double a, b, c;
  std::memcpy(&a, &ina, sizeof(double));
  std::memcpy(&b, &inb, sizeof(double));
  std::memcpy(&c, &inc, sizeof(double));
  double result = 0.0;

  uint8_t fcsr = 0;

  int original_rm = std::fegetround();

  switch (rm) {
    case 0b000: std::fesetround(FE_TONEAREST);
      break;  // RNE
    case 0b001: std::fesetround(FE_TOWARDZERO);
      break; // RTZ
    case 0b010: std::fesetround(FE_DOWNWARD);
      break;   // RDN
    case 0b011: std::fesetround(FE_UPWARD);
      break;     // RUP
      // 0b100 RMM, unsupported
    default: break;
  }

  std::feclearexcept(FE_ALL_EXCEPT);

  switch (op) {
    case AluOp::kAdd: {
      auto sa = static_cast<int64_t>(ina);
      auto sb = static_cast<int64_t>(inb);
      int64_t res = sa + sb;
      // bool overflow = __builtin_add_overflow(sa, sb, &res); // TODO: check this
      return {static_cast<uint64_t>(res), 0};
    }
    case AluOp::FMADD_D: {
      result = std::fma(a, b, c);
      break;
    }
    case AluOp::FMSUB_D: {
      result = std::fma(a, b, -c);
      break;
    }
    case AluOp::FNMADD_D: {
      result = std::fma(-a, b, -c);
      break;
    }
    case AluOp::FNMSUB_D: {
      result = std::fma(-a, b, c);
      break;
    }
    case AluOp::FADD_D: {
      result = a + b;
      break;
    }
    case AluOp::FSUB_D: {
      result = a - b;
      break;
    }
    case AluOp::FMUL_D: {
      result = a * b;
      break;
    }
    case AluOp::FDIV_D: {
      if (b == 0.0) {
        result = std::numeric_limits<double>::quiet_NaN();
        fcsr |= FCSR_DIV_BY_ZERO;
      } else {
        result = a / b;
      }
      break;
    }
    case AluOp::FSQRT_D: {
      if (a < 0.0) {
        result = std::numeric_limits<double>::quiet_NaN();
        fcsr |= FCSR_INVALID_OP;
      } else {
        result = std::sqrt(a);
      }
      break;
    }
    case AluOp::FCVT_W_D: {
      if (!std::isfinite(a) || a > static_cast<double>(INT32_MAX) || a < static_cast<double>(INT32_MIN)) {
        fcsr |= FCSR_INVALID_OP;
        fesetround(original_rm);
        int32_t saturate = (a < 0.0) ? INT32_MIN : INT32_MAX;
        auto res = static_cast<int64_t>(saturate); // sign-extend to XLEN
        return {static_cast<uint64_t>(res), fcsr};
      } else {
        auto ires = static_cast<int32_t>(std::nearbyint(a));
        auto res = static_cast<int64_t>(ires); // sign-extend to XLEN
        fesetround(original_rm);
        return {static_cast<uint64_t>(res), fcsr};
      }
      break;
    }
    case AluOp::FCVT_WU_D: {
      if (!std::isfinite(a) || a > static_cast<double>(UINT32_MAX) || a < 0.0) {
        fcsr |= FCSR_INVALID_OP;
        fesetround(original_rm);
        uint32_t saturate = (a < 0.0) ? 0 : UINT32_MAX;
        auto res = static_cast<int64_t>(static_cast<int32_t>(saturate)); // sign-extend per spec
        return {static_cast<uint64_t>(res), fcsr};
      } else {
        auto ires = static_cast<uint32_t>(std::nearbyint(a));
        auto res = static_cast<int64_t>(static_cast<int32_t>(ires)); // sign-extend
        fesetround(original_rm);
        return {static_cast<uint64_t>(res), fcsr};
      }
      break;
    }
    case AluOp::FCVT_L_D: {
      if (!std::isfinite(a) || a > static_cast<double>(INT64_MAX) || a < static_cast<double>(INT64_MIN)) {
        fcsr |= FCSR_INVALID_OP;
        fesetround(original_rm);
        int64_t saturate = (a < 0.0) ? INT64_MIN : INT64_MAX;
        return {static_cast<uint64_t>(saturate), fcsr};
      } else {
        auto ires = static_cast<int64_t>(std::nearbyint(a));
        fesetround(original_rm);
        return {static_cast<uint64_t>(ires), fcsr};
      }
      break;
    }
    case AluOp::FCVT_LU_D: {
      if (!std::isfinite(a) || a > static_cast<double>(UINT64_MAX) || a < 0.0) {
        fcsr |= FCSR_INVALID_OP;
        fesetround(original_rm);
        uint64_t saturate = (a < 0.0) ? 0 : UINT64_MAX;
        return {saturate, fcsr};
      } else {
        auto ires = static_cast<uint64_t>(std::nearbyint(a));
        fesetround(original_rm);
        return {ires, fcsr};
      }
      break;
    }
    case AluOp::FCVT_D_W: {
      auto ia = static_cast<int32_t>(ina);
      result = static_cast<double>(ia);
      break;

    }
    case AluOp::FCVT_D_WU: {
      auto ua = static_cast<uint32_t>(ina);
      result = static_cast<double>(ua);
      break;
    }
    case AluOp::FCVT_D_L: {
      auto la = static_cast<int64_t>(ina);
      result = static_cast<double>(la);
      break;

    }
    case AluOp::FCVT_S_LU: {
      auto ula = static_cast<uint64_t>(ina);
      result = static_cast<double>(ula);
      break;
    }
    case AluOp::FSGNJ_D: {
      auto a_bits = static_cast<uint64_t>(ina);
      auto b_bits = static_cast<uint64_t>(inb);
      uint64_t temp = (a_bits & 0x7FFFFFFFFFFFFFFF) | (b_bits & 0x8000000000000000);
      std::memcpy(&result, &temp, sizeof(double));
      break;
    }
    case AluOp::FSGNJN_D: {
      auto a_bits = static_cast<uint64_t>(ina);
      auto b_bits = static_cast<uint64_t>(inb);
      uint64_t temp = (a_bits & 0x7FFFFFFFFFFFFFFF) | (~b_bits & 0x8000000000000000);
      std::memcpy(&result, &temp, sizeof(double));
      break;
    }
    case AluOp::FSGNJX_D: {
      auto a_bits = static_cast<uint64_t>(ina);
      auto b_bits = static_cast<uint64_t>(inb);
      uint64_t temp = (a_bits & 0x7FFFFFFFFFFFFFFF) | ((a_bits ^ b_bits) & 0x8000000000000000);
      std::memcpy(&result, &temp, sizeof(double));
      break;
    }
    case AluOp::FMIN_D: {
      if (std::isnan(a) && !std::isnan(b)) {
        result = b;
      } else if (!std::isnan(a) && std::isnan(b)) {
        result = a;
      } else if (std::signbit(a)!=std::signbit(b) && a==b) {
        result = -0.0; // Both zero but with different signs — return -0.0
      } else {
        result = std::fmin(a, b);
      }
      break;
    }
    case AluOp::FMAX_D: {
      if (std::isnan(a) && !std::isnan(b)) {
        result = b;
      } else if (!std::isnan(a) && std::isnan(b)) {
        result = a;
      } else if (std::signbit(a)!=std::signbit(b) && a==b) {
        result = 0.0; // Both zero but with different signs — return +0.0
      } else {
        result = std::fmax(a, b);
      }
      break;
    }
    case AluOp::FEQ_D: {
      if (std::isnan(a) || std::isnan(b)) {
        result = 0.0;
      } else {
        result = (a==b) ? 1.0 : 0.0;
        if (result == 1.0) {
          return {0b1,fcsr};
        }
      }
      break;
    }
    case AluOp::FLT_D: {
      if (std::isnan(a) || std::isnan(b)) {
        result = 0.0;
      } else {
        result = (a < b) ? 1.0 : 0.0;
        if (result == 1.0) {
          return {0b1,fcsr};
        }
      }
      break;
    }
    case AluOp::FLE_D: {
      if (std::isnan(a) || std::isnan(b)) {
        result = 0.0;
      } else {
        result = (a <= b) ? 1.0 : 0.0;
        if (result == 1.0) {
          return {0b1,fcsr};
        }
      }
      break;
    }
    case AluOp::FCLASS_D: {
      uint64_t a_bits = ina;
      double af;
      std::memcpy(&af, &a_bits, sizeof(double));
      uint16_t res = 0;

      if (std::signbit(af) && std::isinf(af)) res |= 1 << 0; // -inf
      else if (std::signbit(af) && std::fpclassify(af)==FP_NORMAL) res |= 1 << 1; // -normal
      else if (std::signbit(af) && std::fpclassify(af)==FP_SUBNORMAL) res |= 1 << 2; // -subnormal
      else if (std::signbit(af) && af==0.0) res |= 1 << 3; // -zero
      else if (!std::signbit(af) && af==0.0) res |= 1 << 4; // +zero
      else if (!std::signbit(af) && std::fpclassify(af)==FP_SUBNORMAL) res |= 1 << 5; // +subnormal
      else if (!std::signbit(af) && std::fpclassify(af)==FP_NORMAL) res |= 1 << 6; // +normal
      else if (!std::signbit(af) && std::isinf(af)) res |= 1 << 7; // +inf
      else if (std::isnan(af) && (a_bits & 0x0008000000000000)==0) res |= 1 << 8; // signaling NaN
      else if (std::isnan(af)) res |= 1 << 9; // quiet NaN

      std::fesetround(original_rm);
      return {res, fcsr};
    }
    case AluOp::FCVT_D_S: {
      auto fa = static_cast<float>(ina);
      result = static_cast<double>(fa);
      break;
    }
    case AluOp::FCVT_S_D: {
      auto da = static_cast<double>(ina);
      result = static_cast<float>(da);
      break;
    }
    case AluOp::FMV_D_X: {
      uint64_t double_bits;
      std::memcpy(&double_bits, &ina, sizeof(double));
      return {double_bits, fcsr};
    }
    case AluOp::FMV_X_D: {
      uint64_t int_bits = ina & 0xFFFFFFFFFFFFFFFF;
      double out;
      std::memcpy(&out, &int_bits, sizeof(double));
      result = out;
      break;
    }
    default: break;
  }

  int raised = std::fetestexcept(FE_ALL_EXCEPT);
  if (raised & FE_INVALID) fcsr |= FCSR_INVALID_OP;
  if (raised & FE_DIVBYZERO) fcsr |= FCSR_DIV_BY_ZERO;
  if (raised & FE_OVERFLOW) fcsr |= FCSR_OVERFLOW;
  if (raised & FE_UNDERFLOW) fcsr |= FCSR_UNDERFLOW;
  if (raised & FE_INEXACT) fcsr |= FCSR_INEXACT;

  std::fesetround(original_rm);

  uint64_t result_bits = 0;
  std::memcpy(&result_bits, &result, sizeof(result));
  return {result_bits, fcsr};
}

void Alu::setFlags(bool carry, bool zero, bool negative, bool overflow) {
  carry_ = carry;
  zero_ = zero;
  negative_ = negative;
  overflow_ = overflow;
}

} // namespace alu