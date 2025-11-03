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
      int64_t sr1 = static_cast<int64_t>(sa1) + static_cast<int64_t>(sb1);
      int64_t sr2 = static_cast<int64_t>(sa2) + static_cast<int64_t>(sb2);

    // Saturate sr1
     if (sr1 > INT32_MAX){ 
      sr1 = INT32_MAX;
     }
     else if (sr1 < INT32_MIN){ 
      sr1 = INT32_MIN;
     }
    // Saturate sr2
     if (sr2 > INT32_MAX){ 
      sr2 = INT32_MAX;
     }
     else if (sr2 < INT32_MIN){ 
      sr2 = INT32_MIN;
     }
     int64_t sr = (sr2 & 0xFFFFFFFFLL) | (sr1 << 32);
     return {sr, false};

    }
    case AluOp::kSub_simd32: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      auto sa1 = static_cast<int32_t>(sa >> 32); // upper 32
      auto sa2 = static_cast<int32_t>(sa - (sa1 << 32)); // lower 32
      auto sb1 = static_cast<int32_t>(sb >> 32); // upper 32
      auto sb2 = static_cast<int32_t>(sb - (sb1 << 32)); // lower 32
      int64_t sr1 = static_cast<int64_t>(sa1) - static_cast<int64_t>(sb1);
      int64_t sr2 = static_cast<int64_t>(sa2) - static_cast<int64_t>(sb2);

    // Saturate sr1
     if (sr1 > INT32_MAX){ 
      sr1 = INT32_MAX;
     }
     else if (sr1 < INT32_MIN){ 
      sr1 = INT32_MIN;
     }
    // Saturate sr2
     if (sr2 > INT32_MAX){ 
      sr2 = INT32_MAX;
     }
     else if (sr2 < INT32_MIN){ 
      sr2 = INT32_MIN;
     }
     int64_t sr = (sr2 & 0xFFFFFFFFLL) | (sr1 << 32);
     return {sr, false};
    }
    case AluOp::kMul_simd32: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      auto sa1 = static_cast<int32_t>(sa >> 32); // upper 32
      auto sa2 = static_cast<int32_t>(sa - (sa1 << 32)); // lower 32
      auto sb1 = static_cast<int32_t>(sb >> 32); // upper 32
      auto sb2 = static_cast<int32_t>(sb - (sb1 << 32)); // lower 32
      int64_t sr1 = static_cast<int64_t>(sa1) * static_cast<int64_t>(sb1);
      int64_t sr2 = static_cast<int64_t>(sa2) * static_cast<int64_t>(sb2);

    // Saturate sr1
     if (sr1 > INT32_MAX){ 
      sr1 = INT32_MAX;
     }
     else if (sr1 < INT32_MIN){ 
      sr1 = INT32_MIN;
     }
    // Saturate sr2
     if (sr2 > INT32_MAX){ 
      sr2 = INT32_MAX;
     }
     else if (sr2 < INT32_MIN){ 
      sr2 = INT32_MIN;
     }
     int64_t sr = (sr2 & 0xFFFFFFFFLL) | (sr1 << 32);
     return {sr, false};

    }
    case AluOp::kLoad_simd32: {
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      auto sa1 = static_cast<int64_t>(sa << 32); // upper 32
      auto sb1 = static_cast<int32_t> (sb); // lower 32
      auto sr1 = sa1 ;
      auto sr2 = sb1 ;
      int64_t sr = sr2 + sr1 ;
      return{sr,false};

    }
    case AluOp::kDiv_simd32: {
      if (b==0) {
        return {0, false};
      }
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      auto sa1 = static_cast<int32_t>(sa >> 32); // upper 32
      auto sa2 = static_cast<int32_t>(sa - (sa1 << 32)); // lower 32
      auto sb1 = static_cast<int32_t>(sb >> 32); // upper 32
      auto sb2 = static_cast<int32_t>(sb - (sb1 << 32)); // lower 32
      int64_t sr1 = static_cast<int64_t>(sa1) / static_cast<int64_t>(sb1);
      int64_t sr2 = static_cast<int64_t>(sa2) / static_cast<int64_t>(sb2);

    // Saturate sr1
     if (sr1 > INT32_MAX){ 
      sr1 = INT32_MAX;
     }
     else if (sr1 < INT32_MIN){ 
      sr1 = INT32_MIN;
     }
    // Saturate sr2
     if (sr2 > INT32_MAX){ 
      sr2 = INT32_MAX;
     }
     else if (sr2 < INT32_MIN){ 
      sr2 = INT32_MIN;
     }
     int64_t sr = (sr2 & 0xFFFFFFFFLL) | (sr1 << 32);
     return {sr, false};

    }
    case AluOp::kRem_simd32: {
      if (b==0) {
        return {0, false};
      }
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      auto sa1 = static_cast<int32_t>(sa >> 32); // upper 32
      auto sa2 = static_cast<int32_t>(sa - (sa1 << 32)); // lower 32
      auto sb1 = static_cast<int32_t>(sb >> 32); // upper 32
      auto sb2 = static_cast<int32_t>(sb - (sb1 << 32)); // lower 32
      int64_t sr1 = static_cast<int64_t>(sa1) % static_cast<int64_t>(sb1);
      int64_t sr2 = static_cast<int64_t>(sa2) % static_cast<int64_t>(sb2);

    // Saturate sr1
     if (sr1 > INT32_MAX){ 
      sr1 = INT32_MAX;
     }
     else if (sr1 < INT32_MIN){ 
      sr1 = INT32_MIN;
     }
    // Saturate sr2
     if (sr2 > INT32_MAX){ 
      sr2 = INT32_MAX;
     }
     else if (sr2 < INT32_MIN){ 
      sr2 = INT32_MIN;
     }
     int64_t sr = (sr2 & 0xFFFFFFFFLL) | (sr1 << 32);
     return {sr, false};

    }
    case AluOp::kAdd_simd16:{
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      
      // Extract 4 16-bit lanes from a
      auto sa1 = static_cast<int16_t>(sa >> 48);
      auto sa_rem48 = sa - (static_cast<int64_t>(sa1) << 48);
      auto sa2 = static_cast<int16_t>(sa_rem48 >> 32);
      auto sa_rem32 = sa_rem48 - (static_cast<int64_t>(sa2) << 32);
      auto sa3 = static_cast<int16_t>(sa_rem32 >> 16);
      auto sa4 = static_cast<int16_t>(sa_rem32 - (static_cast<int64_t>(sa3) << 16));
      
     
      auto sb1 = static_cast<int16_t>(sb >> 48);
      auto sb_rem48 = sb - (static_cast<int64_t>(sb1) << 48);
      auto sb2 = static_cast<int16_t>(sb_rem48 >> 32);
      auto sb_rem32 = sb_rem48 - (static_cast<int64_t>(sb2) << 32);
      auto sb3 = static_cast<int16_t>(sb_rem32 >> 16);
      auto sb4 = static_cast<int16_t>(sb_rem32 - (static_cast<int64_t>(sb3) << 16));
      
      
      int32_t sr1 = static_cast<int32_t>(sa1) + static_cast<int32_t>(sb1);
      int32_t sr2 = static_cast<int32_t>(sa2) + static_cast<int32_t>(sb2);
      int32_t sr3 = static_cast<int32_t>(sa3) + static_cast<int32_t>(sb3);
      int32_t sr4 = static_cast<int32_t>(sa4) + static_cast<int32_t>(sb4);
      
      if (sr1 > INT16_MAX) sr1 = INT16_MAX; else if (sr1 < INT16_MIN) sr1 = INT16_MIN;
      if (sr2 > INT16_MAX) sr2 = INT16_MAX; else if (sr2 < INT16_MIN) sr2 = INT16_MIN;
      if (sr3 > INT16_MAX) sr3 = INT16_MAX; else if (sr3 < INT16_MIN) sr3 = INT16_MIN;
      if (sr4 > INT16_MAX) sr4 = INT16_MAX; else if (sr4 < INT16_MIN) sr4 = INT16_MIN;

      
      int64_t sr = (static_cast<int64_t>(sr1) << 48) |
                   ((static_cast<int64_t>(sr2) & 0xFFFFLL) << 32) |
                   ((static_cast<int64_t>(sr3) & 0xFFFFLL) << 16) |
                   (static_cast<int64_t>(sr4) & 0xFFFFLL);
      return {sr, false};
    }
    case AluOp::kSub_simd16:{
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      
      auto sa1 = static_cast<int16_t>(sa >> 48);
      auto sa_rem48 = sa - (static_cast<int64_t>(sa1) << 48);
      auto sa2 = static_cast<int16_t>(sa_rem48 >> 32);
      auto sa_rem32 = sa_rem48 - (static_cast<int64_t>(sa2) << 32);
      auto sa3 = static_cast<int16_t>(sa_rem32 >> 16);
      auto sa4 = static_cast<int16_t>(sa_rem32 - (static_cast<int64_t>(sa3) << 16));
      
      auto sb1 = static_cast<int16_t>(sb >> 48);
      auto sb_rem48 = sb - (static_cast<int64_t>(sb1) << 48);
      auto sb2 = static_cast<int16_t>(sb_rem48 >> 32);
      auto sb_rem32 = sb_rem48 - (static_cast<int64_t>(sb2) << 32);
      auto sb3 = static_cast<int16_t>(sb_rem32 >> 16);
      auto sb4 = static_cast<int16_t>(sb_rem32 - (static_cast<int64_t>(sb3) << 16));
      
      int32_t sr1 = static_cast<int32_t>(sa1) - static_cast<int32_t>(sb1);
      int32_t sr2 = static_cast<int32_t>(sa2) - static_cast<int32_t>(sb2);
      int32_t sr3 = static_cast<int32_t>(sa3) - static_cast<int32_t>(sb3);
      int32_t sr4 = static_cast<int32_t>(sa4) - static_cast<int32_t>(sb4);
      
      if (sr1 > INT16_MAX) sr1 = INT16_MAX; else if (sr1 < INT16_MIN) sr1 = INT16_MIN;
      if (sr2 > INT16_MAX) sr2 = INT16_MAX; else if (sr2 < INT16_MIN) sr2 = INT16_MIN;
      if (sr3 > INT16_MAX) sr3 = INT16_MAX; else if (sr3 < INT16_MIN) sr3 = INT16_MIN;
      if (sr4 > INT16_MAX) sr4 = INT16_MAX; else if (sr4 < INT16_MIN) sr4 = INT16_MIN;

      int64_t sr = (static_cast<int64_t>(sr1) << 48) |
                   ((static_cast<int64_t>(sr2) & 0xFFFFLL) << 32) |
                   ((static_cast<int64_t>(sr3) & 0xFFFFLL) << 16) |
                   (static_cast<int64_t>(sr4) & 0xFFFFLL);
      return {sr, false};
    }
    case AluOp::kMul_simd16:{
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      
      auto sa1 = static_cast<int16_t>(sa >> 48);
      auto sa_rem48 = sa - (static_cast<int64_t>(sa1) << 48);
      auto sa2 = static_cast<int16_t>(sa_rem48 >> 32);
      auto sa_rem32 = sa_rem48 - (static_cast<int64_t>(sa2) << 32);
      auto sa3 = static_cast<int16_t>(sa_rem32 >> 16);
      auto sa4 = static_cast<int16_t>(sa_rem32 - (static_cast<int64_t>(sa3) << 16));
      
      auto sb1 = static_cast<int16_t>(sb >> 48);
      auto sb_rem48 = sb - (static_cast<int64_t>(sb1) << 48);
      auto sb2 = static_cast<int16_t>(sb_rem48 >> 32);
      auto sb_rem32 = sb_rem48 - (static_cast<int64_t>(sb2) << 32);
      auto sb3 = static_cast<int16_t>(sb_rem32 >> 16);
      auto sb4 = static_cast<int16_t>(sb_rem32 - (static_cast<int64_t>(sb3) << 16));
      
      int32_t sr1 = static_cast<int32_t>(sa1) * static_cast<int32_t>(sb1);
      int32_t sr2 = static_cast<int32_t>(sa2) * static_cast<int32_t>(sb2);
      int32_t sr3 = static_cast<int32_t>(sa3) * static_cast<int32_t>(sb3);
      int32_t sr4 = static_cast<int32_t>(sa4) * static_cast<int32_t>(sb4);
      
      if (sr1 > INT16_MAX) sr1 = INT16_MAX; else if (sr1 < INT16_MIN) sr1 = INT16_MIN;
      if (sr2 > INT16_MAX) sr2 = INT16_MAX; else if (sr2 < INT16_MIN) sr2 = INT16_MIN;
      if (sr3 > INT16_MAX) sr3 = INT16_MAX; else if (sr3 < INT16_MIN) sr3 = INT16_MIN;
      if (sr4 > INT16_MAX) sr4 = INT16_MAX; else if (sr4 < INT16_MIN) sr4 = INT16_MIN;

      int64_t sr = (static_cast<int64_t>(sr1) << 48) |
                   ((static_cast<int64_t>(sr2) & 0xFFFFLL) << 32) |
                   ((static_cast<int64_t>(sr3) & 0xFFFFLL) << 16) |
                   (static_cast<int64_t>(sr4) & 0xFFFFLL);
      return {sr, false};
    }
    case AluOp::kLoad_simd16:{
       
        return {0, false};
    }
    case AluOp::kDiv_simd16:{
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      
      auto sa1 = static_cast<int16_t>(sa >> 48);
      auto sa_rem48 = sa - (static_cast<int64_t>(sa1) << 48);
      auto sa2 = static_cast<int16_t>(sa_rem48 >> 32);
      auto sa_rem32 = sa_rem48 - (static_cast<int64_t>(sa2) << 32);
      auto sa3 = static_cast<int16_t>(sa_rem32 >> 16);
      auto sa4 = static_cast<int16_t>(sa_rem32 - (static_cast<int64_t>(sa3) << 16));
      
      auto sb1 = static_cast<int16_t>(sb >> 48);
      auto sb_rem48 = sb - (static_cast<int64_t>(sb1) << 48);
      auto sb2 = static_cast<int16_t>(sb_rem48 >> 32);
      auto sb_rem32 = sb_rem48 - (static_cast<int64_t>(sb2) << 32);
      auto sb3 = static_cast<int16_t>(sb_rem32 >> 16);
      auto sb4 = static_cast<int16_t>(sb_rem32 - (static_cast<int64_t>(sb3) << 16));
      
      // Per-lane check for division by zero
      int32_t sr1 = (sb1 == 0) ? 0 : (static_cast<int32_t>(sa1) / static_cast<int32_t>(sb1));
      int32_t sr2 = (sb2 == 0) ? 0 : (static_cast<int32_t>(sa2) / static_cast<int32_t>(sb2));
      int32_t sr3 = (sb3 == 0) ? 0 : (static_cast<int32_t>(sa3) / static_cast<int32_t>(sb3));
      int32_t sr4 = (sb4 == 0) ? 0 : (static_cast<int32_t>(sa4) / static_cast<int32_t>(sb4));
      
      if (sr1 > INT16_MAX) sr1 = INT16_MAX; else if (sr1 < INT16_MIN) sr1 = INT16_MIN;
      if (sr2 > INT16_MAX) sr2 = INT16_MAX; else if (sr2 < INT16_MIN) sr2 = INT16_MIN;
      if (sr3 > INT16_MAX) sr3 = INT16_MAX; else if (sr3 < INT16_MIN) sr3 = INT16_MIN;
      if (sr4 > INT16_MAX) sr4 = INT16_MAX; else if (sr4 < INT16_MIN) sr4 = INT16_MIN;

      int64_t sr = (static_cast<int64_t>(sr1) << 48) |
                   ((static_cast<int64_t>(sr2) & 0xFFFFLL) << 32) |
                   ((static_cast<int64_t>(sr3) & 0xFFFFLL) << 16) |
                   (static_cast<int64_t>(sr4) & 0xFFFFLL);
      return {sr, false};
    }
    case AluOp::kRem_simd16:{
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);
      
      auto sa1 = static_cast<int16_t>(sa >> 48);
      auto sa_rem48 = sa - (static_cast<int64_t>(sa1) << 48);
      auto sa2 = static_cast<int16_t>(sa_rem48 >> 32);
      auto sa_rem32 = sa_rem48 - (static_cast<int64_t>(sa2) << 32);
      auto sa3 = static_cast<int16_t>(sa_rem32 >> 16);
      auto sa4 = static_cast<int16_t>(sa_rem32 - (static_cast<int64_t>(sa3) << 16));
      
      auto sb1 = static_cast<int16_t>(sb >> 48);
      auto sb_rem48 = sb - (static_cast<int64_t>(sb1) << 48);
      auto sb2 = static_cast<int16_t>(sb_rem48 >> 32);
      auto sb_rem32 = sb_rem48 - (static_cast<int64_t>(sb2) << 32);
      auto sb3 = static_cast<int16_t>(sb_rem32 >> 16);
      auto sb4 = static_cast<int16_t>(sb_rem32 - (static_cast<int64_t>(sb3) << 16));
      
      // Per-lane check for division by zero
      int32_t sr1 = (sb1 == 0) ? 0 : (static_cast<int32_t>(sa1) % static_cast<int32_t>(sb1));
      int32_t sr2 = (sb2 == 0) ? 0 : (static_cast<int32_t>(sa2) % static_cast<int32_t>(sb2));
      int32_t sr3 = (sb3 == 0) ? 0 : (static_cast<int32_t>(sa3) % static_cast<int32_t>(sb3));
      int32_t sr4 = (sb4 == 0) ? 0 : (static_cast<int32_t>(sa4) % static_cast<int32_t>(sb4));
      
      if (sr1 > INT16_MAX) sr1 = INT16_MAX; else if (sr1 < INT16_MIN) sr1 = INT16_MIN;
      if (sr2 > INT16_MAX) sr2 = INT16_MAX; else if (sr2 < INT16_MIN) sr2 = INT16_MIN;
      if (sr3 > INT16_MAX) sr3 = INT16_MAX; else if (sr3 < INT16_MIN) sr3 = INT16_MIN;
      if (sr4 > INT16_MAX) sr4 = INT16_MAX; else if (sr4 < INT16_MIN) sr4 = INT16_MIN;

      int64_t sr = (static_cast<int64_t>(sr1) << 48) |
                   ((static_cast<int64_t>(sr2) & 0xFFFFLL) << 32) |
                   ((static_cast<int64_t>(sr3) & 0xFFFFLL) << 16) |
                   (static_cast<int64_t>(sr4) & 0xFFFFLL);
      return {sr, false};
    }
   
    
    case AluOp::kAdd_simd8:{
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);

      // Extract 8 8-bit lanes from a
      auto sa1 = static_cast<int8_t>(sa >> 56);
      auto sa_rem56 = sa - (static_cast<int64_t>(sa1) << 56);
      auto sa2 = static_cast<int8_t>(sa_rem56 >> 48);
      auto sa_rem48 = sa_rem56 - (static_cast<int64_t>(sa2) << 48);
      auto sa3 = static_cast<int8_t>(sa_rem48 >> 40);
      auto sa_rem40 = sa_rem48 - (static_cast<int64_t>(sa3) << 40);
      auto sa4 = static_cast<int8_t>(sa_rem40 >> 32);
      auto sa_rem32 = sa_rem40 - (static_cast<int64_t>(sa4) << 32);
      auto sa5 = static_cast<int8_t>(sa_rem32 >> 24);
      auto sa_rem24 = sa_rem32 - (static_cast<int64_t>(sa5) << 24);
      auto sa6 = static_cast<int8_t>(sa_rem24 >> 16);
      auto sa_rem16 = sa_rem24 - (static_cast<int64_t>(sa6) << 16);
      auto sa7 = static_cast<int8_t>(sa_rem16 >> 8);
      auto sa8 = static_cast<int8_t>(sa_rem16 - (static_cast<int64_t>(sa7) << 8));

      // Extract 8 8-bit lanes from b
      auto sb1 = static_cast<int8_t>(sb >> 56);
      auto sb_rem56 = sb - (static_cast<int64_t>(sb1) << 56);
      auto sb2 = static_cast<int8_t>(sb_rem56 >> 48);
      auto sb_rem48 = sb_rem56 - (static_cast<int64_t>(sb2) << 48);
      auto sb3 = static_cast<int8_t>(sb_rem48 >> 40);
      auto sb_rem40 = sb_rem48 - (static_cast<int64_t>(sb3) << 40);
      auto sb4 = static_cast<int8_t>(sb_rem40 >> 32);
      auto sb_rem32 = sb_rem40 - (static_cast<int64_t>(sb4) << 32);
      auto sb5 = static_cast<int8_t>(sb_rem32 >> 24);
      auto sb_rem24 = sb_rem32 - (static_cast<int64_t>(sb5) << 24);
      auto sb6 = static_cast<int8_t>(sb_rem24 >> 16);
      auto sb_rem16 = sb_rem24 - (static_cast<int64_t>(sb6) << 16);
      auto sb7 = static_cast<int8_t>(sb_rem16 >> 8);
      auto sb8 = static_cast<int8_t>(sb_rem16 - (static_cast<int64_t>(sb7) << 8));
      
      // Perform operations, using int16_t for intermediate results
      int16_t sr1 = static_cast<int16_t>(sa1) + static_cast<int16_t>(sb1);
      int16_t sr2 = static_cast<int16_t>(sa2) + static_cast<int16_t>(sb2);
      int16_t sr3 = static_cast<int16_t>(sa3) + static_cast<int16_t>(sb3);
      int16_t sr4 = static_cast<int16_t>(sa4) + static_cast<int16_t>(sb4);
      int16_t sr5 = static_cast<int16_t>(sa5) + static_cast<int16_t>(sb5);
      int16_t sr6 = static_cast<int16_t>(sa6) + static_cast<int16_t>(sb6);
      int16_t sr7 = static_cast<int16_t>(sa7) + static_cast<int16_t>(sb7);
      int16_t sr8 = static_cast<int16_t>(sa8) + static_cast<int16_t>(sb8);

      // Saturate results to 8-bit range
      if (sr1 > INT8_MAX) sr1 = INT8_MAX; else if (sr1 < INT8_MIN) sr1 = INT8_MIN;
      if (sr2 > INT8_MAX) sr2 = INT8_MAX; else if (sr2 < INT8_MIN) sr2 = INT8_MIN;
      if (sr3 > INT8_MAX) sr3 = INT8_MAX; else if (sr3 < INT8_MIN) sr3 = INT8_MIN;
      if (sr4 > INT8_MAX) sr4 = INT8_MAX; else if (sr4 < INT8_MIN) sr4 = INT8_MIN;
      if (sr5 > INT8_MAX) sr5 = INT8_MAX; else if (sr5 < INT8_MIN) sr5 = INT8_MIN;
      if (sr6 > INT8_MAX) sr6 = INT8_MAX; else if (sr6 < INT8_MIN) sr6 = INT8_MIN;
      if (sr7 > INT8_MAX) sr7 = INT8_MAX; else if (sr7 < INT8_MIN) sr7 = INT8_MIN;
      if (sr8 > INT8_MAX) sr8 = INT8_MAX; else if (sr8 < INT8_MIN) sr8 = INT8_MIN;

      // Pack results back into int64_t
      int64_t sr = (static_cast<int64_t>(sr1) << 56) |
                   ((static_cast<int64_t>(sr2) & 0xFFLL) << 48) |
                   ((static_cast<int64_t>(sr3) & 0xFFLL) << 40) |
                   ((static_cast<int64_t>(sr4) & 0xFFLL) << 32) |
                   ((static_cast<int64_t>(sr5) & 0xFFLL) << 24) |
                   ((static_cast<int64_t>(sr6) & 0xFFLL) << 16) |
                   ((static_cast<int64_t>(sr7) & 0xFFLL) << 8)  |
                   (static_cast<int64_t>(sr8) & 0xFFLL);
      return {sr, false};
    }
    case AluOp::kSub_simd8:{
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);

      auto sa1 = static_cast<int8_t>(sa >> 56);
      auto sa_rem56 = sa - (static_cast<int64_t>(sa1) << 56);
      auto sa2 = static_cast<int8_t>(sa_rem56 >> 48);
      auto sa_rem48 = sa_rem56 - (static_cast<int64_t>(sa2) << 48);
      auto sa3 = static_cast<int8_t>(sa_rem48 >> 40);
      auto sa_rem40 = sa_rem48 - (static_cast<int64_t>(sa3) << 40);
      auto sa4 = static_cast<int8_t>(sa_rem40 >> 32);
      auto sa_rem32 = sa_rem40 - (static_cast<int64_t>(sa4) << 32);
      auto sa5 = static_cast<int8_t>(sa_rem32 >> 24);
      auto sa_rem24 = sa_rem32 - (static_cast<int64_t>(sa5) << 24);
      auto sa6 = static_cast<int8_t>(sa_rem24 >> 16);
      auto sa_rem16 = sa_rem24 - (static_cast<int64_t>(sa6) << 16);
      auto sa7 = static_cast<int8_t>(sa_rem16 >> 8);
      auto sa8 = static_cast<int8_t>(sa_rem16 - (static_cast<int64_t>(sa7) << 8));

      auto sb1 = static_cast<int8_t>(sb >> 56);
      auto sb_rem56 = sb - (static_cast<int64_t>(sb1) << 56);
      auto sb2 = static_cast<int8_t>(sb_rem56 >> 48);
      auto sb_rem48 = sb_rem56 - (static_cast<int64_t>(sb2) << 48);
      auto sb3 = static_cast<int8_t>(sb_rem48 >> 40);
      auto sb_rem40 = sb_rem48 - (static_cast<int64_t>(sb3) << 40);
      auto sb4 = static_cast<int8_t>(sb_rem40 >> 32);
      auto sb_rem32 = sb_rem40 - (static_cast<int64_t>(sb4) << 32);
      auto sb5 = static_cast<int8_t>(sb_rem32 >> 24);
      auto sb_rem24 = sb_rem32 - (static_cast<int64_t>(sb5) << 24);
      auto sb6 = static_cast<int8_t>(sb_rem24 >> 16);
      auto sb_rem16 = sb_rem24 - (static_cast<int64_t>(sb6) << 16);
      auto sb7 = static_cast<int8_t>(sb_rem16 >> 8);
      auto sb8 = static_cast<int8_t>(sb_rem16 - (static_cast<int64_t>(sb7) << 8));
      
      int16_t sr1 = static_cast<int16_t>(sa1) - static_cast<int16_t>(sb1);
      int16_t sr2 = static_cast<int16_t>(sa2) - static_cast<int16_t>(sb2);
      int16_t sr3 = static_cast<int16_t>(sa3) - static_cast<int16_t>(sb3);
      int16_t sr4 = static_cast<int16_t>(sa4) - static_cast<int16_t>(sb4);
      int16_t sr5 = static_cast<int16_t>(sa5) - static_cast<int16_t>(sb5);
      int16_t sr6 = static_cast<int16_t>(sa6) - static_cast<int16_t>(sb6);
      int16_t sr7 = static_cast<int16_t>(sa7) - static_cast<int16_t>(sb7);
      int16_t sr8 = static_cast<int16_t>(sa8) - static_cast<int16_t>(sb8);

      if (sr1 > INT8_MAX) sr1 = INT8_MAX; else if (sr1 < INT8_MIN) sr1 = INT8_MIN;
      if (sr2 > INT8_MAX) sr2 = INT8_MAX; else if (sr2 < INT8_MIN) sr2 = INT8_MIN;
      if (sr3 > INT8_MAX) sr3 = INT8_MAX; else if (sr3 < INT8_MIN) sr3 = INT8_MIN;
      if (sr4 > INT8_MAX) sr4 = INT8_MAX; else if (sr4 < INT8_MIN) sr4 = INT8_MIN;
      if (sr5 > INT8_MAX) sr5 = INT8_MAX; else if (sr5 < INT8_MIN) sr5 = INT8_MIN;
      if (sr6 > INT8_MAX) sr6 = INT8_MAX; else if (sr6 < INT8_MIN) sr6 = INT8_MIN;
      if (sr7 > INT8_MAX) sr7 = INT8_MAX; else if (sr7 < INT8_MIN) sr7 = INT8_MIN;
      if (sr8 > INT8_MAX) sr8 = INT8_MAX; else if (sr8 < INT8_MIN) sr8 = INT8_MIN;

      int64_t sr = (static_cast<int64_t>(sr1) << 56) |
                   ((static_cast<int64_t>(sr2) & 0xFFLL) << 48) |
                   ((static_cast<int64_t>(sr3) & 0xFFLL) << 40) |
                   ((static_cast<int64_t>(sr4) & 0xFFLL) << 32) |
                   ((static_cast<int64_t>(sr5) & 0xFFLL) << 24) |
                   ((static_cast<int64_t>(sr6) & 0xFFLL) << 16) |
                   ((static_cast<int64_t>(sr7) & 0xFFLL) << 8)  |
                   (static_cast<int64_t>(sr8) & 0xFFLL);
      return {sr, false};
    }
    case AluOp::kMul_simd8:{
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);

      auto sa1 = static_cast<int8_t>(sa >> 56);
      auto sa_rem56 = sa - (static_cast<int64_t>(sa1) << 56);
      auto sa2 = static_cast<int8_t>(sa_rem56 >> 48);
      auto sa_rem48 = sa_rem56 - (static_cast<int64_t>(sa2) << 48);
      auto sa3 = static_cast<int8_t>(sa_rem48 >> 40);
      auto sa_rem40 = sa_rem48 - (static_cast<int64_t>(sa3) << 40);
      auto sa4 = static_cast<int8_t>(sa_rem40 >> 32);
      auto sa_rem32 = sa_rem40 - (static_cast<int64_t>(sa4) << 32);
      auto sa5 = static_cast<int8_t>(sa_rem32 >> 24);
      auto sa_rem24 = sa_rem32 - (static_cast<int64_t>(sa5) << 24);
      auto sa6 = static_cast<int8_t>(sa_rem24 >> 16);
      auto sa_rem16 = sa_rem24 - (static_cast<int64_t>(sa6) << 16);
      auto sa7 = static_cast<int8_t>(sa_rem16 >> 8);
      auto sa8 = static_cast<int8_t>(sa_rem16 - (static_cast<int64_t>(sa7) << 8));

      auto sb1 = static_cast<int8_t>(sb >> 56);
      auto sb_rem56 = sb - (static_cast<int64_t>(sb1) << 56);
      auto sb2 = static_cast<int8_t>(sb_rem56 >> 48);
      auto sb_rem48 = sb_rem56 - (static_cast<int64_t>(sb2) << 48);
      auto sb3 = static_cast<int8_t>(sb_rem48 >> 40);
      auto sb_rem40 = sb_rem48 - (static_cast<int64_t>(sb3) << 40);
      auto sb4 = static_cast<int8_t>(sb_rem40 >> 32);
      auto sb_rem32 = sb_rem40 - (static_cast<int64_t>(sb4) << 32);
      auto sb5 = static_cast<int8_t>(sb_rem32 >> 24);
      auto sb_rem24 = sb_rem32 - (static_cast<int64_t>(sb5) << 24);
      auto sb6 = static_cast<int8_t>(sb_rem24 >> 16);
      auto sb_rem16 = sb_rem24 - (static_cast<int64_t>(sb6) << 16);
      auto sb7 = static_cast<int8_t>(sb_rem16 >> 8);
      auto sb8 = static_cast<int8_t>(sb_rem16 - (static_cast<int64_t>(sb7) << 8));
      
      int16_t sr1 = static_cast<int16_t>(sa1) * static_cast<int16_t>(sb1);
      int16_t sr2 = static_cast<int16_t>(sa2) * static_cast<int16_t>(sb2);
      int16_t sr3 = static_cast<int16_t>(sa3) * static_cast<int16_t>(sb3);
      int16_t sr4 = static_cast<int16_t>(sa4) * static_cast<int16_t>(sb4);
      int16_t sr5 = static_cast<int16_t>(sa5) * static_cast<int16_t>(sb5);
      int16_t sr6 = static_cast<int16_t>(sa6) * static_cast<int16_t>(sb6);
      int16_t sr7 = static_cast<int16_t>(sa7) * static_cast<int16_t>(sb7);
      int16_t sr8 = static_cast<int16_t>(sa8) * static_cast<int16_t>(sb8);

      if (sr1 > INT8_MAX) sr1 = INT8_MAX; else if (sr1 < INT8_MIN) sr1 = INT8_MIN;
      if (sr2 > INT8_MAX) sr2 = INT8_MAX; else if (sr2 < INT8_MIN) sr2 = INT8_MIN;
      if (sr3 > INT8_MAX) sr3 = INT8_MAX; else if (sr3 < INT8_MIN) sr3 = INT8_MIN;
      if (sr4 > INT8_MAX) sr4 = INT8_MAX; else if (sr4 < INT8_MIN) sr4 = INT8_MIN;
      if (sr5 > INT8_MAX) sr5 = INT8_MAX; else if (sr5 < INT8_MIN) sr5 = INT8_MIN;
      if (sr6 > INT8_MAX) sr6 = INT8_MAX; else if (sr6 < INT8_MIN) sr6 = INT8_MIN;
      if (sr7 > INT8_MAX) sr7 = INT8_MAX; else if (sr7 < INT8_MIN) sr7 = INT8_MIN;
      if (sr8 > INT8_MAX) sr8 = INT8_MAX; else if (sr8 < INT8_MIN) sr8 = INT8_MIN;

      int64_t sr = (static_cast<int64_t>(sr1) << 56) |
                   ((static_cast<int64_t>(sr2) & 0xFFLL) << 48) |
                   ((static_cast<int64_t>(sr3) & 0xFFLL) << 40) |
                   ((static_cast<int64_t>(sr4) & 0xFFLL) << 32) |
                   ((static_cast<int64_t>(sr5) & 0xFFLL) << 24) |
                   ((static_cast<int64_t>(sr6) & 0xFFLL) << 16) |
                   ((static_cast<int64_t>(sr7) & 0xFFLL) << 8)  |
                   (static_cast<int64_t>(sr8) & 0xFFLL);
      return {sr, false};
    }
    case AluOp::kLoad_simd8:{
        // As requested, left empty
        return {0, false};
    }
    case AluOp::kDiv_simd8:{
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);

      auto sa1 = static_cast<int8_t>(sa >> 56);
      auto sa_rem56 = sa - (static_cast<int64_t>(sa1) << 56);
      auto sa2 = static_cast<int8_t>(sa_rem56 >> 48);
      auto sa_rem48 = sa_rem56 - (static_cast<int64_t>(sa2) << 48);
      auto sa3 = static_cast<int8_t>(sa_rem48 >> 40);
      auto sa_rem40 = sa_rem48 - (static_cast<int64_t>(sa3) << 40);
      auto sa4 = static_cast<int8_t>(sa_rem40 >> 32);
      auto sa_rem32 = sa_rem40 - (static_cast<int64_t>(sa4) << 32);
      auto sa5 = static_cast<int8_t>(sa_rem32 >> 24);
      auto sa_rem24 = sa_rem32 - (static_cast<int64_t>(sa5) << 24);
      auto sa6 = static_cast<int8_t>(sa_rem24 >> 16);
      auto sa_rem16 = sa_rem24 - (static_cast<int64_t>(sa6) << 16);
      auto sa7 = static_cast<int8_t>(sa_rem16 >> 8);
      auto sa8 = static_cast<int8_t>(sa_rem16 - (static_cast<int64_t>(sa7) << 8));

      auto sb1 = static_cast<int8_t>(sb >> 56);
      auto sb_rem56 = sb - (static_cast<int64_t>(sb1) << 56);
      auto sb2 = static_cast<int8_t>(sb_rem56 >> 48);
      auto sb_rem48 = sb_rem56 - (static_cast<int64_t>(sb2) << 48);
      auto sb3 = static_cast<int8_t>(sb_rem48 >> 40);
      auto sb_rem40 = sb_rem48 - (static_cast<int64_t>(sb3) << 40);
      auto sb4 = static_cast<int8_t>(sb_rem40 >> 32);
      auto sb_rem32 = sb_rem40 - (static_cast<int64_t>(sb4) << 32);
      auto sb5 = static_cast<int8_t>(sb_rem32 >> 24);
      auto sb_rem24 = sb_rem32 - (static_cast<int64_t>(sb5) << 24);
      auto sb6 = static_cast<int8_t>(sb_rem24 >> 16);
      auto sb_rem16 = sb_rem24 - (static_cast<int64_t>(sb6) << 16);
      auto sb7 = static_cast<int8_t>(sb_rem16 >> 8);
      auto sb8 = static_cast<int8_t>(sb_rem16 - (static_cast<int64_t>(sb7) << 8));
      
      // Per-lane check for division by zero
      int16_t sr1 = (sb1 == 0) ? 0 : (static_cast<int16_t>(sa1) / static_cast<int16_t>(sb1));
      int16_t sr2 = (sb2 == 0) ? 0 : (static_cast<int16_t>(sa2) / static_cast<int16_t>(sb2));
      int16_t sr3 = (sb3 == 0) ? 0 : (static_cast<int16_t>(sa3) / static_cast<int16_t>(sb3));
      int16_t sr4 = (sb4 == 0) ? 0 : (static_cast<int16_t>(sa4) / static_cast<int16_t>(sb4));
      int16_t sr5 = (sb5 == 0) ? 0 : (static_cast<int16_t>(sa5) / static_cast<int16_t>(sb5));
      int16_t sr6 = (sb6 == 0) ? 0 : (static_cast<int16_t>(sa6) / static_cast<int16_t>(sb6));
      int16_t sr7 = (sb7 == 0) ? 0 : (static_cast<int16_t>(sa7) / static_cast<int16_t>(sb7));
      int16_t sr8 = (sb8 == 0) ? 0 : (static_cast<int16_t>(sa8) / static_cast<int16_t>(sb8));

      if (sr1 > INT8_MAX) sr1 = INT8_MAX; else if (sr1 < INT8_MIN) sr1 = INT8_MIN;
      if (sr2 > INT8_MAX) sr2 = INT8_MAX; else if (sr2 < INT8_MIN) sr2 = INT8_MIN;
      if (sr3 > INT8_MAX) sr3 = INT8_MAX; else if (sr3 < INT8_MIN) sr3 = INT8_MIN;
      if (sr4 > INT8_MAX) sr4 = INT8_MAX; else if (sr4 < INT8_MIN) sr4 = INT8_MIN;
      if (sr5 > INT8_MAX) sr5 = INT8_MAX; else if (sr5 < INT8_MIN) sr5 = INT8_MIN;
      if (sr6 > INT8_MAX) sr6 = INT8_MAX; else if (sr6 < INT8_MIN) sr6 = INT8_MIN;
      if (sr7 > INT8_MAX) sr7 = INT8_MAX; else if (sr7 < INT8_MIN) sr7 = INT8_MIN;
      if (sr8 > INT8_MAX) sr8 = INT8_MAX; else if (sr8 < INT8_MIN) sr8 = INT8_MIN;

      int64_t sr = (static_cast<int64_t>(sr1) << 56) |
                   ((static_cast<int64_t>(sr2) & 0xFFLL) << 48) |
                   ((static_cast<int64_t>(sr3) & 0xFFLL) << 40) |
                   ((static_cast<int64_t>(sr4) & 0xFFLL) << 32) |
                   ((static_cast<int64_t>(sr5) & 0xFFLL) << 24) |
                   ((static_cast<int64_t>(sr6) & 0xFFLL) << 16) |
                   ((static_cast<int64_t>(sr7) & 0xFFLL) << 8)  |
                   (static_cast<int64_t>(sr8) & 0xFFLL);
      return {sr, false};
    }
    case AluOp::kRem_simd8:{
      auto sa = static_cast<int64_t>(a);
      auto sb = static_cast<int64_t>(b);

      auto sa1 = static_cast<int8_t>(sa >> 56);
      auto sa_rem56 = sa - (static_cast<int64_t>(sa1) << 56);
      auto sa2 = static_cast<int8_t>(sa_rem56 >> 48);
      auto sa_rem48 = sa_rem56 - (static_cast<int64_t>(sa2) << 48);
      auto sa3 = static_cast<int8_t>(sa_rem48 >> 40);
      auto sa_rem40 = sa_rem48 - (static_cast<int64_t>(sa3) << 40);
      auto sa4 = static_cast<int8_t>(sa_rem40 >> 32);
      auto sa_rem32 = sa_rem40 - (static_cast<int64_t>(sa4) << 32);
      auto sa5 = static_cast<int8_t>(sa_rem32 >> 24);
      auto sa_rem24 = sa_rem32 - (static_cast<int64_t>(sa5) << 24);
      auto sa6 = static_cast<int8_t>(sa_rem24 >> 16);
      auto sa_rem16 = sa_rem24 - (static_cast<int64_t>(sa6) << 16);
      auto sa7 = static_cast<int8_t>(sa_rem16 >> 8);
      auto sa8 = static_cast<int8_t>(sa_rem16 - (static_cast<int64_t>(sa7) << 8));

      auto sb1 = static_cast<int8_t>(sb >> 56);
      auto sb_rem56 = sb - (static_cast<int64_t>(sb1) << 56);
      auto sb2 = static_cast<int8_t>(sb_rem56 >> 48);
      auto sb_rem48 = sb_rem56 - (static_cast<int64_t>(sb2) << 48);
      auto sb3 = static_cast<int8_t>(sb_rem48 >> 40);
      auto sb_rem40 = sb_rem48 - (static_cast<int64_t>(sb3) << 40);
      auto sb4 = static_cast<int8_t>(sb_rem40 >> 32);
      auto sb_rem32 = sb_rem40 - (static_cast<int64_t>(sb4) << 32);
      auto sb5 = static_cast<int8_t>(sb_rem32 >> 24);
      auto sb_rem24 = sb_rem32 - (static_cast<int64_t>(sb5) << 24);
      auto sb6 = static_cast<int8_t>(sb_rem24 >> 16);
      auto sb_rem16 = sb_rem24 - (static_cast<int64_t>(sb6) << 16);
      auto sb7 = static_cast<int8_t>(sb_rem16 >> 8);
      auto sb8 = static_cast<int8_t>(sb_rem16 - (static_cast<int64_t>(sb7) << 8));
      
      // Per-lane check for division by zero
      int16_t sr1 = (sb1 == 0) ? 0 : (static_cast<int16_t>(sa1) % static_cast<int16_t>(sb1));
      int16_t sr2 = (sb2 == 0) ? 0 : (static_cast<int16_t>(sa2) % static_cast<int16_t>(sb2));
      int16_t sr3 = (sb3 == 0) ? 0 : (static_cast<int16_t>(sa3) % static_cast<int16_t>(sb3));
      int16_t sr4 = (sb4 == 0) ? 0 : (static_cast<int16_t>(sa4) % static_cast<int16_t>(sb4));
      int16_t sr5 = (sb5 == 0) ? 0 : (static_cast<int16_t>(sa5) % static_cast<int16_t>(sb5));
      int16_t sr6 = (sb6 == 0) ? 0 : (static_cast<int16_t>(sa6) % static_cast<int16_t>(sb6));
      int16_t sr7 = (sb7 == 0) ? 0 : (static_cast<int16_t>(sa7) % static_cast<int16_t>(sb7));
      int16_t sr8 = (sb8 == 0) ? 0 : (static_cast<int16_t>(sa8) % static_cast<int16_t>(sb8));

      if (sr1 > INT8_MAX) sr1 = INT8_MAX; else if (sr1 < INT8_MIN) sr1 = INT8_MIN;
      if (sr2 > INT8_MAX) sr2 = INT8_MAX; else if (sr2 < INT8_MIN) sr2 = INT8_MIN;
      if (sr3 > INT8_MAX) sr3 = INT8_MAX; else if (sr3 < INT8_MIN) sr3 = INT8_MIN;
      if (sr4 > INT8_MAX) sr4 = INT8_MAX; else if (sr4 < INT8_MIN) sr4 = INT8_MIN;
      if (sr5 > INT8_MAX) sr5 = INT8_MAX; else if (sr5 < INT8_MIN) sr5 = INT8_MIN;
      if (sr6 > INT8_MAX) sr6 = INT8_MAX; else if (sr6 < INT8_MIN) sr6 = INT8_MIN;
      if (sr7 > INT8_MAX) sr7 = INT8_MAX; else if (sr7 < INT8_MIN) sr7 = INT8_MIN;
      if (sr8 > INT8_MAX) sr8 = INT8_MAX; else if (sr8 < INT8_MIN) sr8 = INT8_MIN;

      int64_t sr = (static_cast<int64_t>(sr1) << 56) |
                   ((static_cast<int64_t>(sr2) & 0xFFLL) << 48) |
                   ((static_cast<int64_t>(sr3) & 0xFFLL) << 40) |
                   ((static_cast<int64_t>(sr4) & 0xFFLL) << 32) |
                   ((static_cast<int64_t>(sr5) & 0xFFLL) << 24) |
                   ((static_cast<int64_t>(sr6) & 0xFFLL) << 16) |
                   ((static_cast<int64_t>(sr7) & 0xFFLL) << 8)  |
                   (static_cast<int64_t>(sr8) & 0xFFLL);
      return {sr, false};
    }
    case AluOp::kAdd_simd4: {
    int64_t res = 0;
    for (int i = 0; i < 16; i++) {
        int64_t laneA = (a >> (i * 4)) & 0xF;
        int64_t laneB = (b >> (i * 4)) & 0xF;
        int64_t sum = laneA + laneB;
        if (sum > 15) sum = 7; 
        else if(sum < -8) sum = -8;  // saturate
        res |= (sum & 0xF) << (i * 4);
    }
    return {res,false};
    }
    case AluOp::kSub_simd4: {
     int64_t res = 0;
    for (int i = 0; i < 16; i++) {
        int64_t laneA = (a >> (i * 4)) & 0xF;
        int64_t laneB = (b >> (i * 4)) & 0xF;
        int64_t sum = laneA - laneB;
        if (sum > 7) sum = 7;  
        else if(sum < -8) sum = -8; // saturate
        res |= (sum & 0xF) << (i * 4);
    }
    return {res,false};
    }
    case AluOp::kMul_simd4: {
     int64_t res = 0;
    for (int i = 0; i < 16; i++) {
        int64_t laneA = (a >> (i * 4)) & 0xF;
        int64_t laneB = (b >> (i * 4)) & 0xF;
        int64_t sum = laneA * laneB;
        if (sum > 7) sum = 7; 
        else if(sum < -8) sum = -8;  // saturate
        res |= (sum & 0xF) << (i * 4);
    }
    return {res,false};
    }
    case AluOp::kLoad_simd4: {
     //left empty for now 
      int64_t res = 0;
      return {res,false};
    }
    case AluOp::kDiv_simd4: {
      int64_t res = 0;
     for (int i = 0; i < 16; i++) {
        int64_t laneA = (a >> (i * 4)) & 0xF;
        int64_t laneB = (b >> (i * 4)) & 0xF;
        int64_t sum = laneA / laneB;
        if (sum > 7) sum = 7; 
        else if(sum < -8) sum = -8;  // saturate
        res |= (sum & 0xF) << (i * 4);
    }
    return {res,false};
    }
    case AluOp::kRem_simd4: {
     int64_t res = 0;
     for (int i = 0; i < 16; i++) {
        int64_t laneA = (a >> (i * 4)) & 0xF;
        int64_t laneB = (b >> (i * 4)) & 0xF;
        int64_t sum = laneA % laneB;
        if (sum > 7) sum = 7; 
        else if(sum < -8) sum = -8; // saturate
        res |= (sum & 0xF) << (i * 4);
    }
    return {res,false};
    }
    case AluOp::kAdd_simd2: {
     int64_t res = 0;
    for (int i = 0; i < 32; i++) {
        int64_t laneA = (a >> (i * 2)) & 0x3;
        int64_t laneB = (b >> (i * 2)) & 0x3;
        int64_t sum = laneA + laneB;
        if (sum > 1) sum = 1;
        else if(sum < -2) sum =-2;  // saturate
        res |= (sum & 0x3) << (i * 2);
    }
    return {res,false};
    }
    case AluOp::kSub_simd2: {
     int64_t res = 0;
    for (int i = 0; i < 32; i++) {
        int64_t laneA = (a >> (i * 2)) & 0x3;
        int64_t laneB = (b >> (i * 2)) & 0x3;
        int64_t sum = laneA - laneB;
        if (sum > 1) sum = 1;  // saturate
        else if(sum < -2) sum =-2;
        res |= (sum & 0x3) << (i * 2);
    }
    return {res,false};
    }
    case AluOp::kMul_simd2: {
     int64_t res = 0;
    for (int i = 0; i < 32; i++) {
        int64_t laneA = (a >> (i * 2)) & 0x3;
        int64_t laneB = (b >> (i * 2)) & 0x3;
        int64_t sum = laneA * laneB;
        if (sum > 1) sum = 1;  // saturate
        else if(sum < -2) sum =-2;
        res |= (sum & 0x3) << (i * 2);
    }
    return {res,false};
    }
    case AluOp::kLoad_simd2: {
     int64_t res = 0;
     return {res,false};
    }
    case AluOp::kDiv_simd2: {
     int64_t res = 0;
    for (int i = 0; i < 32; i++) {
        int64_t laneA = (a >> (i * 2)) & 0x3;
        int64_t laneB = (b >> (i * 2)) & 0x3;
        int64_t sum = laneA / laneB;
        if (sum > 1) sum = 1;  // saturate
        else if(sum < -2) sum =-2;
        res |= (sum & 0x3) << (i * 2);
    }
    return {res,false};
    }
    case AluOp::kRem_simd2: {
     int64_t res = 0;
    for (int i = 0; i < 32; i++) {
        int64_t laneA = (a >> (i * 2)) & 0x3;
        int64_t laneB = (b >> (i * 2)) & 0x3;
        int64_t sum = laneA % laneB;
        if (sum > 1) sum = 1;  // saturate
        else if(sum < -2) sum =-2;
        res |= (sum & 0x3) << (i * 2);
    }
    return {res,false};
    }
    case AluOp::kAdd_simdb: {
          
    }
    case AluOp::kSub_simdb: {
     
    }
    case AluOp::kMul_simdb: {
     
    }
    case AluOp::kLoad_simdb: {
     
    }
    case AluOp::kDiv_simdb: {
     
    }
    case AluOp::kRem_simdb: {
     
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