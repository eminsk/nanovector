; =============================================================================
; NanoVector — High-Performance AVX2+FMA Vector Engine DLL (x86-64 FASM)
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; MIT License
; =============================================================================

format PE64 GUI 6.0 DLL
entry DllEntryPoint

include 'C:\proekts\FASM\INCLUDE\WIN64A.INC'

section '.text' code readable executable

proc DllEntryPoint hinstDLL, fdwReason, lpvReserved
    mov eax, 1
    ret
endp

; -----------------------------------------------------------------------------
; const char* nanovec_simd_isa(void)
; -----------------------------------------------------------------------------
align 16
nanovec_simd_isa:
    lea rax, [isa_str]
    ret

; Include microkernels
include 'nanovector64_kernel.inc'

section '.data' data readable
isa_str db 'AVX2+FMA (FASM x86-64, 256-bit SIMD)', 0

section '.edata' export data readable
export 'nanovector64.dll',\
       nanovec_fasm_dot,         'nanovec_fasm_dot',\
       nanovec_fasm_norm_sq,     'nanovec_fasm_norm_sq',\
       nanovec_fasm_l2_sq,       'nanovec_fasm_l2_sq',\
       nanovec_fasm_normalize,   'nanovec_fasm_normalize',\
       nanovec_fasm_batch_dot,   'nanovec_fasm_batch_dot',\
       nanovec_fasm_batch_l2_sq, 'nanovec_fasm_batch_l2_sq',\
       nanovec_fasm_batch_cosine,'nanovec_fasm_batch_cosine',\
       nanovec_simd_isa,         'nanovec_simd_isa',\
       nanovec_simd_isa,         'nanovec_fasm_simd_isa'

section '.reloc' fixups data readable discardable
if $=$$
    dd 0,8
end if
