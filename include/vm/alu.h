/**
 * @file alu.h
 * @brief Contains the definition of the alu class for performing arithmetic and logic operations.
 * @author Vishank Singh, httpa://github.com/VishankSingh
 */
#ifndef ALU_H
#define ALU_H

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <ostream>

// #pragma float_control(precise, on)
// #pragma STDC FENV_ACCESS ON

#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#pragma GCC optimize ("no-fast-math") 

#define FCSR_INVALID_OP   (1 << 0)  // Invalid operation
#define FCSR_DIV_BY_ZERO  (1 << 1)  // Divide by zero
#define FCSR_OVERFLOW     (1 << 2)  // Overflow
#define FCSR_UNDERFLOW    (1 << 3)  // Underflow
#define FCSR_INEXACT      (1 << 4)  // Inexact result

namespace alu {

enum class AluOp {
    kNone, ///< No operation.
    kAdd, ///< Addition operation.
    kAddw, ///< Addition word operation.
    kSub, ///< Subtraction operation.
    kSubw, ///< Subtraction word operation.
    kMul, ///< Multiplication operation.
    kMulh, ///< Multiplication high operation.
    kMulhsu, ///< Multiplication high signed and unsigned operation.
    kMulhu, ///< Multiplication high unsigned operation.
    kMulw, ///< Multiplication word operation.
    kDiv, ///< Division operation.
    kDivw, ///< Division word operation.
    kDivu, ///< Unsigned division operation.
    kDivuw, ///< Unsigned division word operation.
    kRem, ///< Remainder operation.
    kRemw, ///< Remainder word operation.
    kRemu, ///< Unsigned remainder operation.
    kRemuw, ///< Unsigned remainder word operation.
    kAnd, ///< Bitwise kAnd operation.
    kOr, ///< Bitwise kOr operation.
    kXor, ///< Bitwise kXor operation.

    kAdd_simd32, ///<Newly added instruction>///
    kSub_simd32,
    kMul_simd32,
    kLoad_simd32,
    kDiv_simd32,
    kRem_simd32,

    kAdd_simd16, ///<Newly added instruction>///
    kSub_simd16,
    kMul_simd16,
    kLoad_simd16,
    kDiv_simd16,
    kRem_simd16,

    kAdd_simd8, ///<Newly added instruction>///
    kSub_simd8,
    kMul_simd8,
    kLoad_simd8,
    kDiv_simd8,
    kRem_simd8,
    

    kAdd_simd4, ///<Newly added instruction>///
    kSub_simd4,
    kMul_simd4,
    kLoad_simd4,
    kDiv_simd4,
    kRem_simd4,


    kAdd_simd2, ///<Newly added instruction>///
    kSub_simd2,
    kMul_simd2,
    kLoad_simd2,
    kDiv_simd2,
    kRem_simd2,


    kAdd_simdb, ///<Newly added instruction>///
    kSub_simdb,
    kMul_simdb,
    kLoad_simdb,
    kDiv_simdb,
    kRem_simdb,

    kSll, ///< Shift left logical operation.
    kSllw, ///< Shift left logical word operation.
    kSrl, ///< Shift right logical operation.
    kSrlw, ///< Shift right logical word operation.
    kSra, ///< Shift right arithmetic operation.
    kSraw, ///< Shift right arithmetic word operation.
    kSlt, ///< Set less than operation.
    kSltu, ///< Unsigned set less than operation.

    // Floating point operations
    kFmadd_s, ///< Floating point multiply-add single operation.
    kFmsub_s, ///< Floating point multiply-subtract single operation.
    kFnmadd_s, ///< Floating point negative multiply-add single operation.
    kFnmsub_s, ///< Floating point negative multiply-subtract single operation.

    FADD_S, ///< Floating point addition operation.
    FSUB_S, ///< Floating point subtraction operation.
    FMUL_S, ///< Floating point multiplication operation.
    FDIV_S, ///< Floating point division operation.
    FSQRT_S, ///< Floating point square root operation.
    FSGNJ_S, ///< Floating point sign inject operation.
    FSGNJN_S, ///< Floating point sign inject negative operation.
    FSGNJX_S, ///< Floating point sign inject kXor operation.
    FMIN_S, ///< Floating point minimum operation.
    FMAX_S, ///< Floating point maximum operation.
    FEQ_S, ///< Floating point equal operation.
    FLT_S, ///< Floating point less than operation.
    FLE_S, ///< Floating point less than or equal operation.
    FCLASS_S, ///< Floating point class operation.

    FCVT_W_S, ///< Floating point convert to word operation.
    FCVT_WU_S, ///< Floating point convert to unsigned word operation.
    FCVT_L_S, ///< Floating point convert double to long operation.
    FCVT_LU_S, ///< Floating point convert double to unsigned long operation.
    
    FCVT_S_W, ///< Floating point convert word to operation.
    FCVT_S_WU, ///< Floating point convert unsigned word to operation.
    FCVT_S_L, ///< Floating point convert long to single operation.
    FCVT_S_LU, ///< Floating point convert unsigned long to single operation.

    FMV_X_W, ///< Floating point move to integer operation.
    FMV_W_X, ///< Floating point move from integer operation.

    FMADD_D, ///< Floating point multiply-add double operation.
    FMSUB_D, ///< Floating point multiply-subtract double operation.
    FNMADD_D, ///< Floating point negative multiply-add double operation.
    FNMSUB_D, ///< Floating point negative multiply-subtract double operation.

    FADD_D, ///< Floating point addition double operation.
    FSUB_D, ///< Floating point subtraction double operation.
    FMUL_D, ///< Floating point multiplication double operation.
    FDIV_D, ///< Floating point division double operation.
    FSQRT_D, ///< Floating point square root double operation.
    FSGNJ_D, ///< Floating point sign inject double operation.
    FSGNJN_D, ///< Floating point sign inject negative double operation.
    FSGNJX_D, ///< Floating point sign inject kXor double operation.
    FMIN_D, ///< Floating point minimum double operation.
    FMAX_D, ///< Floating point maximum double operation.
    FEQ_D, ///< Floating point equal double operation.
    FLT_D, ///< Floating point less than double operation.
    FLE_D, ///< Floating point less than or equal double operation.
    FCLASS_D, ///< Floating point class double operation.

    FCVT_W_D, ///< Floating point convert double to word operation.
    FCVT_WU_D, ///< Floating point convert double to unsigned word operation.
    FCVT_L_D, ///< Floating point convert double to long operation.
    FCVT_LU_D, ///< Floating point convert double to unsigned long operation.

    FCVT_D_W, ///< Floating point convert word to double operation.
    FCVT_D_WU, ///< Floating point convert unsigned word to double operation.
    FCVT_D_L, ///< Floating point convert long to double operation.
    FCVT_D_LU, ///< Floating point convert unsigned long to double operation.

    FCVT_S_D, ///< Floating point convert double to single operation.
    FCVT_D_S, ///< Floating point convert single to double operation.

    FMV_D_X, ///< Floating point move to integer double operation.
    FMV_X_D, ///< Floating point move from integer double operation.

        // BFloat16 Operations
    FADD_BF16,
    FSUB_BF16,
    FMUL_BF16,
    FMAX_BF16,
    FMADD_BF16,
};

inline std::ostream& operator<<(std::ostream& os, const AluOp& op) {
    switch (op) {
        case AluOp::kNone: os << "kNone"; break;
        case AluOp::kAdd: os << "kAdd"; break;
        case AluOp::kSub: os << "kSub"; break;
        case AluOp::kMul: os << "kMul"; break;
        case AluOp::kDiv: os << "kDiv"; break;
        case AluOp::kDivu: os << "kDivu"; break;
        case AluOp::kRem: os << "kRem"; break;
        case AluOp::kRemu: os << "kRemu"; break;
        case AluOp::kAnd: os << "kAnd"; break;
        case AluOp::kOr: os << "kOr"; break;
        case AluOp::kXor: os << "kXor"; break;

        case AluOp::kAdd_simd32: os << "kAdd_simd32";break;
        case AluOp::kSub_simd32: os << "kSub_simd32";break;
        case AluOp::kMul_simd32: os << "kMul_simd32";break;
        case AluOp::kLoad_simd32: os << "kLoad_simd32";break;
        case AluOp::kDiv_simd32: os << "kDiv_simd32";break;
        case AluOp::kRem_simd32: os << "kRem_simd32";break;

        case AluOp::kAdd_simd16: os << "kAdd_simd16";break;
        case AluOp::kSub_simd16: os << "kSub_simd16";break;
        case AluOp::kMul_simd16: os << "kMul_simd16";break;
        case AluOp::kLoad_simd16: os << "kLoad_simd16";break;
        case AluOp::kDiv_simd16: os << "kDiv_simd16";break;
        case AluOp::kRem_simd16: os << "kRem_simd16";break;
        
        case AluOp::kAdd_simd8: os << "kAdd_simd8";break;
        case AluOp::kSub_simd8: os << "kSub_simd8";break;
        case AluOp::kMul_simd8: os << "kMul_simd8";break;
        case AluOp::kLoad_simd8: os << "kLoad_simd8";break;
        case AluOp::kDiv_simd8: os << "kDiv_simd8";break;
        case AluOp::kRem_simd8: os << "kRem_simd8";break;
        
        case AluOp::kAdd_simd4: os << "kAdd_simd4";break;
        case AluOp::kSub_simd4: os << "kSub_simd4";break;
        case AluOp::kMul_simd4: os << "kMul_simd4";break;
        case AluOp::kLoad_simd4: os << "kLoad_simd4";break;
        case AluOp::kDiv_simd4: os << "kDiv_simd4";break;
        case AluOp::kRem_simd4: os << "kRem_simd4";break;

        case AluOp::kAdd_simd2: os << "kAdd_simd2";break;
        case AluOp::kSub_simd2: os << "kSub_simd2";break;
        case AluOp::kMul_simd2: os << "kMul_simd2";break;
        case AluOp::kLoad_simd2: os << "kLoad_simd2";break;
        case AluOp::kDiv_simd2: os << "kDiv_simd2";break;
        case AluOp::kRem_simd2: os << "kRem_simd2";break;

        case AluOp::kAdd_simdb: os << "kAdd_simdb";break;
        case AluOp::kSub_simdb: os << "kSub_simdb";break;
        case AluOp::kMul_simdb: os << "kMul_simdb";break;
        case AluOp::kLoad_simdb: os << "kLoad_simdb";break;
        case AluOp::kDiv_simdb: os << "kDiv_simdb";break;
        case AluOp::kRem_simdb: os << "kRem_simdb";break;


        case AluOp::kSll: os << "kSll"; break;
        case AluOp::kSrl: os << "kSrl"; break;
        case AluOp::kSra: os << "kSra"; break;
        case AluOp::kSlt: os << "kSlt"; break;
        case AluOp::kSltu: os << "kSltu"; break;
        case AluOp::kAddw: os << "kAddw"; break;
        case AluOp::kSubw: os << "kSubw"; break;
        case AluOp::kMulw: os << "kMulw"; break;
        case AluOp::kDivw: os << "kDivw"; break;
        case AluOp::kDivuw: os << "kDivuw"; break;
        case AluOp::kRemw: os << "kRemw"; break;
        case AluOp::kRemuw: os << "kRemuw"; break;
        case AluOp::kMulh: os << "kMulh"; break;
        case AluOp::kMulhsu: os << "kMulhsu"; break;
        case AluOp::kMulhu: os << "kMulhu"; break;
        case AluOp::kSllw: os << "kSllw"; break;
        case AluOp::kSrlw: os << "kSrlw"; break;
        case AluOp::kSraw: os << "kSraw"; break;
        case AluOp::kFmadd_s: os << "kFmadd_s"; break;
        case AluOp::kFmsub_s: os << "kFmsub_s"; break;
        case AluOp::kFnmadd_s: os << "kFnmadd_s"; break;
        case AluOp::kFnmsub_s: os << "kFnmsub_s"; break;
        case AluOp::FADD_S: os << "FADD_S"; break;
        case AluOp::FSUB_S: os << "FSUB_S"; break;
        case AluOp::FMUL_S: os << "FMUL_S"; break;
        case AluOp::FDIV_S: os << "FDIV_S"; break;
        case AluOp::FSQRT_S: os << "FSQRT_S"; break;
        case AluOp::FSGNJ_S: os << "FSGNJ_S"; break;
        case AluOp::FSGNJN_S: os << "FSGNJN_S"; break;
        case AluOp::FSGNJX_S: os << "FSGNJX_S"; break;
        case AluOp::FMIN_S: os << "FMIN_S"; break;
        case AluOp::FMAX_S: os << "FMAX_S"; break;
        case AluOp::FEQ_S: os << "FEQ_S"; break;
        case AluOp::FLT_S: os << "FLT_S"; break;
        case AluOp::FLE_S: os << "FLE_S"; break;
        case AluOp::FCLASS_S: os << "FCLASS_S"; break;
        case AluOp::FCVT_W_S: os << "FCVT_W_S"; break;
        case AluOp::FCVT_WU_S: os << "FCVT_WU_S"; break;
        case AluOp::FCVT_L_S: os << "FCVT_L_S"; break;
        case AluOp::FCVT_LU_S: os << "FCVT_LU_S"; break;
        case AluOp::FCVT_S_W: os << "FCVT_S_W"; break;
        case AluOp::FCVT_S_WU: os << "FCVT_S_WU"; break;
        case AluOp::FCVT_S_L: os << "FCVT_S_L"; break;
        case AluOp::FCVT_S_LU: os << "FCVT_S_LU"; break;
        case AluOp::FMADD_D: os << "FMADD_D"; break;
        case AluOp::FMSUB_D: os << "FMSUB_D"; break;
        case AluOp::FNMADD_D: os << "FNMADD_D"; break;
        case AluOp::FNMSUB_D: os << "FNMSUB_D"; break;
        case AluOp::FADD_D: os << "FADD_D"; break;
        case AluOp::FSUB_D: os << "FSUB_D"; break;
        case AluOp::FMUL_D: os << "FMUL_D"; break;
        case AluOp::FDIV_D: os << "FDIV_D"; break;
        case AluOp::FSQRT_D: os << "FSQRT_D"; break;
        case AluOp::FSGNJ_D: os << "FSGNJ_D"; break;

                // BFloat16 Operations
case AluOp::FADD_BF16: os << "FADD_BF16"; break;
 case AluOp::FSUB_BF16: os << "FSUB_BF16"; break;
 case AluOp::FMUL_BF16: os << "FMUL_BF16"; break;
 case AluOp::FMAX_BF16: os << "FMAX_BF16"; break;
  case AluOp::FMADD_BF16: os << "FMADD_BF16"; break;


        default: os << "UNKNOWN"; break;
    }
    return os;
}
/**
 * @brief The alu class is responsible for performing arithmetic and logic operations.
 */
class Alu {
public:
    bool carry_ = false; ///< Carry flag.
    bool zero_ = false; ///< Zero flag.
    bool negative_ = false; ///< Negative flag.
    bool overflow_ = false; ///< Overflow flag.

    Alu() = default;
    ~Alu() = default;

    /**
     * @brief Executes the given alu operation.
     * @tparam T Integer type (int32_t, uint32_t, etc.).
     * @param op The alu operation.
     * @param a First operand.
     * @param b Second operand.
     * @return A pair (result, overflow_flag).
     */
    [[nodiscard]] static std::pair<uint64_t, bool> execute(AluOp op, uint64_t a, uint64_t b) ;

    // TODO: check all the floating point operations

    [[nodiscard]] static std::pair<uint64_t, uint8_t> fpexecute(AluOp op, uint64_t ina, uint64_t inb, uint64_t inc, uint8_t rm) ;

    [[nodiscard]] static std::pair<uint64_t, bool> dfpexecute(AluOp op, uint64_t ina, uint64_t inb, uint64_t inc, uint8_t rm) ;

    void setFlags(bool carry, bool zero, bool negative, bool overflow);

};
}

#endif // ALU_H