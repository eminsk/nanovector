; =============================================================================
; NanoVector — High-Performance 32-bit SSE2 Vector Engine DLL (x86 FASM)
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; MIT License
; =============================================================================

format PE GUI 4.0 DLL
entry DllEntryPoint

include 'C:\proekts\FASM\INCLUDE\WIN32A.INC'

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
    lea eax, [isa_str]
    ret

; Include 32-bit microkernels
include 'nanovector32_kernel.inc'

section '.data' data readable
isa_str db 'SSE2 (FASM x86 32-bit, 128-bit SIMD)', 0

section '.edata' export data readable
export 'nanovector32.dll',\
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
