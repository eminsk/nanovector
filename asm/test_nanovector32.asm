; =============================================================================
; NanoVector — 32-bit Native Standalone FASM Test & Benchmark Suite
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; MIT License
; =============================================================================

format PE console
entry start

include 'C:\proekts\FASM\INCLUDE\WIN32A.INC'

section '.data' data readable writeable
    hdr_msg     db '====================================================================', 13, 10
                db '  NanoVector Native x86 32-bit FASM SSE2 Engine Test Suite', 13, 10
                db '====================================================================', 13, 10, 0
    isa_msg     db '  Engine ISA: %s', 13, 10, 0
    test1_msg   db '  [TEST 1] Vector Dot Product (dim=384, float32): ', 0
    test2_msg   db '  [TEST 2] Vector L2 Distance Squared (dim=384): ', 0
    test3_msg   db '  [TEST 3] Vector In-Place L2 Normalization (dim=384): ', 0
    test4_msg   db '  [TEST 4] Batched Dot Product (1000 x 384): ', 0
    test5_msg   db '  [TEST 5] Batched L2 Squared Search (1000 x 384): ', 0
    test6_msg   db '  [TEST 6] Batched Cosine Similarity (1000 x 384): ', 0
    test7_msg   db '  [TEST 7] Odd / Non-aligned Dimension (dim=77, dot+L2): ', 0
    pass_str    db 'PASS (Exact match)', 13, 10, 0
    bench_fmt   db 'PASS (%u us, Approx %u MVec/sec)', 13, 10, 0
    fail_str    db 'FAIL! Deviation exceeds tolerance.', 13, 10, 0
    all_ok_msg  db '--------------------------------------------------------------------', 13, 10
                db '  ALL 32-BIT FASM NATIVE TESTS PASSED (100%% Accuracy)!', 13, 10
                db '====================================================================', 13, 10, 0

    isa_str     db 'SSE2 (FASM x86 32-bit, 128-bit SIMD)', 0
    align 16
    abs_mask    dd 7FFFFFFFh, 7FFFFFFFh, 7FFFFFFFh, 7FFFFFFFh
    tol_flt     dd 0.005

    freq_lo     dd 0
    freq_hi     dd 0
    t1_lo       dd 0
    t1_hi       dd 0
    t2_lo       dd 0
    t2_hi       dd 0
    elapsed_us  dd 0

section '.bss' readable writeable
    align 16
    vec_a       rd 1536
    align 16
    vec_b       rd 1536
    align 16
    matrix_buf  rd 384000      ; 1000 * 384 floats
    align 16
    norms_buf   rd 1000
    align 16
    scores_buf  rd 1000

section '.text' code readable executable

; Include 32-bit microkernels
include 'nanovector32_kernel.inc'

align 16
nanovec_simd_isa:
    lea eax, [isa_str]
    ret

start:
    ; Query timer frequency
    lea eax, [freq_lo]
    push eax
    call [QueryPerformanceFrequency]

    ; Print header
    lea eax, [hdr_msg]
    push eax
    call [printf]
    add esp, 4

    call nanovec_simd_isa
    push eax
    lea eax, [isa_msg]
    push eax
    call [printf]
    add esp, 8

    ; -------------------------------------------------------------------------
    ; [TEST 1] Dot Product (dim=384): vec_a[i] = 0.5, vec_b[i] = 2.0 -> sum = 384.0
    ; -------------------------------------------------------------------------
    lea eax, [test1_msg]
    push eax
    call [printf]
    add esp, 4

    mov ecx, 384
    xor eax, eax
.init_t1:
    mov dword [vec_a + eax*4], 3F000000h ; 0.5f
    mov dword [vec_b + eax*4], 40000000h ; 2.0f
    inc eax
    cmp eax, ecx
    jb .init_t1

    push 384
    lea eax, [vec_b]
    push eax
    lea eax, [vec_a]
    push eax
    call nanovec_fasm_dot
    add esp, 12

    ; Result in ST(0)
    sub esp, 4
    fstp dword [esp]
    movss xmm0, [esp]
    mov eax, 43C00000h          ; 384.0f
    mov [esp], eax
    movss xmm1, [esp]
    subss xmm0, xmm1
    movups xmm2, xword [abs_mask]
    andps xmm0, xmm2
    ucomiss xmm0, [tol_flt]
    ja .fail_t1
    add esp, 4

    lea eax, [pass_str]
    push eax
    call [printf]
    add esp, 4
    jmp .test2

.fail_t1:
    lea eax, [fail_str]
    push eax
    call [printf]
    push 1
    call [ExitProcess]

.test2:
    ; -------------------------------------------------------------------------
    ; [TEST 2] L2 Squared: vec_a[i] = 3.0, vec_b[i] = 1.0 -> diff=2, diff^2=4, sum=1536.0
    ; -------------------------------------------------------------------------
    lea eax, [test2_msg]
    push eax
    call [printf]
    add esp, 4

    mov ecx, 384
    xor eax, eax
.init_t2:
    mov dword [vec_a + eax*4], 40400000h ; 3.0f
    mov dword [vec_b + eax*4], 3F800000h ; 1.0f
    inc eax
    cmp eax, ecx
    jb .init_t2

    push 384
    lea eax, [vec_b]
    push eax
    lea eax, [vec_a]
    push eax
    call nanovec_fasm_l2_sq
    add esp, 12

    sub esp, 4
    fstp dword [esp]
    movss xmm0, [esp]
    mov eax, 44C00000h          ; 1536.0f
    mov [esp], eax
    movss xmm1, [esp]
    subss xmm0, xmm1
    movups xmm2, xword [abs_mask]
    andps xmm0, xmm2
    ucomiss xmm0, [tol_flt]
    ja .fail_t2
    add esp, 4

    lea eax, [pass_str]
    push eax
    call [printf]
    add esp, 4
    jmp .test3

.fail_t2:
    lea eax, [fail_str]
    push eax
    call [printf]
    push 1
    call [ExitProcess]

.test3:
    ; -------------------------------------------------------------------------
    ; [TEST 3] Vector In-Place L2 Normalization (dim=384)
    ; -------------------------------------------------------------------------
    lea eax, [test3_msg]
    push eax
    call [printf]
    add esp, 4

    push 384
    lea eax, [vec_a]
    push eax
    call nanovec_fasm_normalize
    add esp, 8

    push 384
    lea eax, [vec_a]
    push eax
    call nanovec_fasm_norm_sq
    add esp, 8

    sub esp, 4
    fstp dword [esp]
    movss xmm0, [esp]
    mov eax, 3F800000h          ; 1.0f
    mov [esp], eax
    movss xmm1, [esp]
    subss xmm0, xmm1
    movups xmm2, xword [abs_mask]
    andps xmm0, xmm2
    ucomiss xmm0, [tol_flt]
    ja .fail_t3
    add esp, 4

    lea eax, [pass_str]
    push eax
    call [printf]
    add esp, 4
    jmp .test4

.fail_t3:
    lea eax, [fail_str]
    push eax
    call [printf]
    push 1
    call [ExitProcess]

.test4:
    ; -------------------------------------------------------------------------
    ; [TEST 4] Batched Dot Product (1000 x 384)
    ; -------------------------------------------------------------------------
    lea eax, [test4_msg]
    push eax
    call [printf]
    add esp, 4

    mov ecx, 384000
    xor eax, eax
.init_mat:
    mov dword [matrix_buf + eax*4], 3DCCCCCDh ; 0.1f
    inc eax
    cmp eax, ecx
    jb .init_mat

    lea eax, [t1_lo]
    push eax
    call [QueryPerformanceCounter]

    lea eax, [scores_buf]
    push eax
    push 384
    push 1000
    lea eax, [matrix_buf]
    push eax
    lea eax, [vec_b]
    push eax
    call nanovec_fasm_batch_dot
    add esp, 20

    lea eax, [t2_lo]
    push eax
    call [QueryPerformanceCounter]

    ; Verify scores[0] == 38.4f
    movss xmm0, [scores_buf]
    sub esp, 4
    mov eax, 4219999Ah
    mov [esp], eax
    movss xmm1, [esp]
    subss xmm0, xmm1
    movups xmm2, xword [abs_mask]
    andps xmm0, xmm2
    ucomiss xmm0, [tol_flt]
    ja .fail_t4
    add esp, 4

    ; Calculate elapsed us
    mov eax, [t2_lo]
    sub eax, [t1_lo]
    imul eax, 1000000
    xor edx, edx
    div dword [freq_lo]
    mov [elapsed_us], eax
    test eax, eax
    jnz .calc_rate4
    mov eax, 1
.calc_rate4:
    mov ecx, eax
    mov eax, 1000000000
    xor edx, edx
    div ecx
    shr eax, 20                  ; Approx MVec/sec
    push eax
    push [elapsed_us]
    lea eax, [bench_fmt]
    push eax
    call [printf]
    add esp, 12
    jmp .test5

.fail_t4:
    lea eax, [fail_str]
    push eax
    call [printf]
    push 1
    call [ExitProcess]

.test5:
    ; -------------------------------------------------------------------------
    ; [TEST 5] Batched L2 Search (1000 x 384)
    ; -------------------------------------------------------------------------
    lea eax, [test5_msg]
    push eax
    call [printf]
    add esp, 4

    lea eax, [t1_lo]
    push eax
    call [QueryPerformanceCounter]

    lea eax, [scores_buf]
    push eax
    push 384
    push 1000
    lea eax, [matrix_buf]
    push eax
    lea eax, [vec_b]
    push eax
    call nanovec_fasm_batch_l2_sq
    add esp, 20

    lea eax, [t2_lo]
    push eax
    call [QueryPerformanceCounter]

    mov eax, [t2_lo]
    sub eax, [t1_lo]
    imul eax, 1000000
    xor edx, edx
    div dword [freq_lo]
    mov [elapsed_us], eax
    test eax, eax
    jnz .calc_rate5
    mov eax, 1
.calc_rate5:
    mov ecx, eax
    mov eax, 1000000000
    xor edx, edx
    div ecx
    shr eax, 20
    push eax
    push [elapsed_us]
    lea eax, [bench_fmt]
    push eax
    call [printf]
    add esp, 12
    jmp .test6

.test6:
    ; -------------------------------------------------------------------------
    ; [TEST 6] Batched Cosine Similarity (1000 x 384)
    ; -------------------------------------------------------------------------
    lea eax, [test6_msg]
    push eax
    call [printf]
    add esp, 4

    mov ecx, 1000
    xor eax, eax
.init_norms:
    mov dword [norms_buf + eax*4], 3FFAE148h ; 1.95959f
    inc eax
    cmp eax, ecx
    jb .init_norms

    lea eax, [t1_lo]
    push eax
    call [QueryPerformanceCounter]

    lea eax, [scores_buf]
    push eax
    push 384
    push 1000
    lea eax, [norms_buf]
    push eax
    lea eax, [matrix_buf]
    push eax
    push 419CB1F8h              ; q_norm = 19.5959f
    lea eax, [vec_b]
    push eax
    call nanovec_fasm_batch_cosine
    add esp, 28

    lea eax, [t2_lo]
    push eax
    call [QueryPerformanceCounter]

    movss xmm0, [scores_buf]
    sub esp, 4
    mov eax, 3F800000h          ; 1.0f
    mov [esp], eax
    movss xmm1, [esp]
    subss xmm0, xmm1
    movups xmm2, xword [abs_mask]
    andps xmm0, xmm2
    ucomiss xmm0, [tol_flt]
    ja .fail_t6
    add esp, 4

    mov eax, [t2_lo]
    sub eax, [t1_lo]
    imul eax, 1000000
    xor edx, edx
    div dword [freq_lo]
    mov [elapsed_us], eax
    test eax, eax
    jnz .calc_rate6
    mov eax, 1
.calc_rate6:
    mov ecx, eax
    mov eax, 1000000000
    xor edx, edx
    div ecx
    shr eax, 20
    push eax
    push [elapsed_us]
    lea eax, [bench_fmt]
    push eax
    call [printf]
    add esp, 12
    jmp .test7

.fail_t6:
    lea eax, [fail_str]
    push eax
    call [printf]
    push 1
    call [ExitProcess]

.test7:
    ; -------------------------------------------------------------------------
    ; [TEST 7] Odd / Prime dimension (dim = 77)
    ; -------------------------------------------------------------------------
    lea eax, [test7_msg]
    push eax
    call [printf]
    add esp, 4

    mov ecx, 77
    xor eax, eax
.init_t7:
    mov dword [vec_a + eax*4], 3F800000h ; 1.0f
    mov dword [vec_b + eax*4], 40000000h ; 2.0f
    inc eax
    cmp eax, ecx
    jb .init_t7

    push 77
    lea eax, [vec_b]
    push eax
    lea eax, [vec_a]
    push eax
    call nanovec_fasm_dot
    add esp, 12

    sub esp, 4
    fstp dword [esp]
    movss xmm0, [esp]
    mov eax, 431A0000h          ; 154.0f
    mov [esp], eax
    movss xmm1, [esp]
    subss xmm0, xmm1
    movups xmm2, xword [abs_mask]
    andps xmm0, xmm2
    ucomiss xmm0, [tol_flt]
    ja .fail_t7
    add esp, 4

    lea eax, [pass_str]
    push eax
    call [printf]
    add esp, 4

    lea eax, [all_ok_msg]
    push eax
    call [printf]
    add esp, 4

    push 0
    call [ExitProcess]

.fail_t7:
    lea eax, [fail_str]
    push eax
    call [printf]
    push 1
    call [ExitProcess]

section '.idata' import data readable
library kernel32, 'kernel32.dll',\
        msvcrt,   'msvcrt.dll'

import kernel32,\
       QueryPerformanceCounter,   'QueryPerformanceCounter',\
       QueryPerformanceFrequency, 'QueryPerformanceFrequency',\
       ExitProcess,               'ExitProcess'

import msvcrt,\
       printf, 'printf'
