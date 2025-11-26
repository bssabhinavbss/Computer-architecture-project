#include <bits/stdc++.h>
using namespace std;

struct SimdResult {
    uint32_t upper_result; 
    uint32_t lower_result; 
};

uint64_t perform_simd_add(uint32_t a, uint32_t b, uint32_t c, uint32_t d){

    uint32_t upper_result = a + b; 
    uint32_t lower_result = c + d;

    uint64_t packed_result = 
        (static_cast<uint64_t>(upper_result) << 32) | 
        static_cast<uint64_t>(lower_result);
    
    return packed_result;
}

bool check_simd_correctness(uint64_t packed_result, uint32_t a, uint32_t b, uint32_t c, uint32_t d){
   
    uint32_t expected_upper = a + b;
    uint32_t expected_lower = c + d;
    
    
    uint32_t actual_upper = static_cast<uint32_t>(packed_result >> 32);
    uint32_t actual_lower = static_cast<uint32_t>(packed_result & 0xFFFFFFFF);
    
   
    bool upper_ok = (actual_upper == expected_upper);
    bool lower_ok = (actual_lower == expected_lower);
    
   
    cout << "Upper Check (a + b):" << endl;
    cout << "  Expected: 0x" << hex << expected_upper << dec << endl;
    cout << "  Actual:   0x" << hex << actual_upper << dec << " (" << (upper_ok ? "CORRECT" : "FAIL") << ")" << endl;

    cout << "Lower Check (c + d):" << endl;
    cout << "  Expected: 0x" << hex << expected_lower << dec << endl;
    cout << "  Actual:   0x" << hex << actual_lower << dec << " (" << (lower_ok ? "CORRECT" : "FAIL") << ")" << endl;
    

    return upper_ok && lower_ok;
}

int main(){
   
    uint32_t A = 0x10000000;
    uint32_t B = 0x20000000;
    uint32_t C = 0xFFFFFFF0; 
    uint32_t D = 0x00000015;

    cout << hex << std::uppercase << std::setfill('0');
    cout << "Inputs:\n";
    cout << "  A: 0x" <<setw(8) << A << "\tB: 0x" <<setw(8) << B << "\n";
    cout << "  C: 0x" <<setw(8) << C << "\tD: 0x" <<setw(8) << D << "\n\n";
    

    uint64_t result = perform_simd_add(A, B, C, D);
    
    cout << "SIMD Packed Result (64-bit):\n";
    cout << "  0x" <<setw(16) << result << "\n\n";

  
    bool is_correct = check_simd_correctness(result, A, B, C, D);

    cout << "\nFinal Result: The SIMD addition is " << (is_correct ? "CORRECT." : "INCORRECT.") << endl;

    return 0;
}
