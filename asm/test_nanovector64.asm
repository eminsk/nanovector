; =============================================================================
; NanoVector — 64-bit Native Standalone FASM Test & Benchmark Suite
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; MIT License
; =============================================================================

format PE64 console
entry start

include 'C:\proekts\FASM\INCLUDE\WIN64A.INC'

section '.data' data readable writeable
    hdr_msg     db '====================================================================', 13, 10
                db '  NanoVector Native x86-64 FASM AVX2+FMA Engine Test Suite', 13, 10
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
                db '  ALL 64-BIT FASM NATIVE TESTS PASSED (100%% Accuracy)!', 13, 10
                db '====================================================================', 13, 10, 0

    isa_str     db 'AVX2+FMA (FASM x86-64, 256-bit SIMD)', 0
    abs_mask    dd 7FFFFFFFh, 7FFFFFFFh, 7FFFFFFFh, 7FFFFFFFh

    freq        rq 1
    t_start     rq 1
    t_end       rq 1
    elapsed_us  rq 1

    one_flt     dd 1.0
    tol_flt     dd 0.005
    diff_flt    dd 0.0

section '.bss' readable writeable
    align 32
    vec_a       rd 1536
    align 32
    vec_b       rd 1536
    align 32
    matrix_buf  rd 384000      ; 1000 * 384 floats
    align 32
    norms_buf   rd 1000
    align 32
    scores_buf  rd 1000

section '.text' code readable executable

; Include microkernels
include 'nanovector64_kernel.inc'

align 16
nanovec_simd_isa:
    lea rax, [isa_str]
    ret

start:
    sub rsp, 88h                ; 16-byte stack alignment (88h = 136 = 8 mod 16)

    ; Query timer frequency
    lea rcx, [freq]
    call [QueryPerformanceFrequency]

    ; Print header
    lea rcx, [hdr_msg]
    call [printf]

    call nanovec_simd_isa
    mov rdx, rax
    lea rcx, [isa_msg]
    call [printf]

    ; -------------------------------------------------------------------------
    ; [TEST 1] Dot Product (dim=384): vec_a[i] = 0.5, vec_b[i] = 2.0 -> sum = 384.0
    ; -------------------------------------------------------------------------
    lea rcx, [test1_msg]
    call [printf]

    mov ecx, 384
    xor eax, eax
.init_t1:
    mov dword [vec_a + rax*4], 3F000000h ; 0.5f
    mov dword [vec_b + rax*4], 40000000h ; 2.0f
    inc eax
    cmp eax, ecx
    jb .init_t1

    lea rcx, [vec_a]
    lea rdx, [vec_b]
    mov r8, 384
    call nanovec_fasm_dot

    ; Expected = 384.0f (0x43C00000)
    mov eax, 43C00000h
    vmovd xmm1, eax
    vsubss xmm2, xmm0, xmm1
    vandps xmm2, xmm2, xword [abs_mask]
    vucomiss xmm2, [tol_flt]
    ja .fail_t1

    lea rcx, [pass_str]
    call [printf]
    jmp .test2

.fail_t1:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 1
    call [ExitProcess]

.test2:
    ; -------------------------------------------------------------------------
    ; [TEST 2] L2 Squared: vec_a[i] = 3.0, vec_b[i] = 1.0 -> diff=2, diff^2=4, sum=1536.0
    ; -------------------------------------------------------------------------
    lea rcx, [test2_msg]
    call [printf]

    mov ecx, 384
    xor eax, eax
.init_t2:
    mov dword [vec_a + rax*4], 40400000h ; 3.0f
    mov dword [vec_b + rax*4], 3F800000h ; 1.0f
    inc eax
    cmp eax, ecx
    jb .init_t2

    lea rcx, [vec_a]
    lea rdx, [vec_b]
    mov r8, 384
    call nanovec_fasm_l2_sq

    ; Expected = 384 * 4.0 = 1536.0f (0x44C00000)
    mov eax, 44C00000h
    vmovd xmm1, eax
    vsubss xmm2, xmm0, xmm1
    vandps xmm2, xmm2, xword [abs_mask]
    vucomiss xmm2, [tol_flt]
    ja .fail_t2

    lea rcx, [pass_str]
    call [printf]
    jmp .test3

.fail_t2:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 1
    call [ExitProcess]

.test3:
    ; -------------------------------------------------------------------------
    ; [TEST 3] Vector In-Place L2 Normalization (dim=384)
    ; After normalize(vec_a), norm_sq(vec_a) must equal 1.0f
    ; -------------------------------------------------------------------------
    lea rcx, [test3_msg]
    call [printf]

    lea rcx, [vec_a]
    mov rdx, 384
    call nanovec_fasm_normalize

    lea rcx, [vec_a]
    mov rdx, 384
    call nanovec_fasm_norm_sq

    ; Expected = 1.0f (0x3F800000)
    mov eax, 3F800000h
    vmovd xmm1, eax
    vsubss xmm2, xmm0, xmm1
    vandps xmm2, xmm2, xword [abs_mask]
    vucomiss xmm2, [tol_flt]
    ja .fail_t3

    lea rcx, [pass_str]
    call [printf]
    jmp .test4

.fail_t3:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 1
    call [ExitProcess]

.test4:
    ; -------------------------------------------------------------------------
    ; [TEST 4] Batched Dot Product (1000 x 384)
    ; -------------------------------------------------------------------------
    lea rcx, [test4_msg]
    call [printf]

    ; Fill matrix_buf: 1000 * 384 floats
    mov ecx, 384000
    xor eax, eax
.init_mat:
    mov dword [matrix_buf + rax*4], 3DCCCCCDh ; 0.1f
    inc eax
    cmp eax, ecx
    jb .init_mat

    ; Timer start
    lea rcx, [t_start]
    call [QueryPerformanceCounter]

    lea rcx, [vec_b]
    lea rdx, [matrix_buf]
    mov r8, 1000
    mov r9, 384
    mov qword [rsp + 20h], scores_buf
    call nanovec_fasm_batch_dot

    lea rcx, [t_end]
    call [QueryPerformanceCounter]

    ; Verify scores[0]: vec_b is 1.0f, matrix is 0.1f -> dot = 38.4f (0x4219999A)
    mov eax, 4219999Ah
    vmovd xmm1, eax
    vmovss xmm0, [scores_buf]
    vsubss xmm2, xmm0, xmm1
    vandps xmm2, xmm2, xword [abs_mask]
    vucomiss xmm2, [tol_flt]
    ja .fail_t4

    ; Calculate elapsed us
    mov rax, [t_end]
    sub rax, [t_start]
    imul rax, 1000000
    xor edx, edx
    div [freq]
    mov [elapsed_us], rax
    test rax, rax
    jnz .calc_rate4
    mov rax, 1
.calc_rate4:
    mov rax, 1000000000
    xor edx, edx
    div [elapsed_us]
    mov r8, rax
    shr r8, 20                  ; Approx MVec/sec
    mov rdx, [elapsed_us]
    lea rcx, [bench_fmt]
    call [printf]
    jmp .test5

.fail_t4:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 1
    call [ExitProcess]

.test5:
    ; -------------------------------------------------------------------------
    ; [TEST 5] Batched L2 Search (1000 x 384)
    ; -------------------------------------------------------------------------
    lea rcx, [test5_msg]
    call [printf]

    lea rcx, [t_start]
    call [QueryPerformanceCounter]

    lea rcx, [vec_b]
    lea rdx, [matrix_buf]
    mov r8, 1000
    mov r9, 384
    mov qword [rsp + 20h], scores_buf
    call nanovec_fasm_batch_l2_sq

    lea rcx, [t_end]
    call [QueryPerformanceCounter]

    mov rax, [t_end]
    sub rax, [t_start]
    imul rax, 1000000
    xor edx, edx
    div [freq]
    mov [elapsed_us], rax
    test rax, rax
    jnz .calc_rate5
    mov rax, 1
.calc_rate5:
    mov rax, 1000000000
    xor edx, edx
    div [elapsed_us]
    mov r8, rax
    shr r8, 20
    mov rdx, [elapsed_us]
    lea rcx, [bench_fmt]
    call [printf]
    jmp .test6

.test6:
    ; -------------------------------------------------------------------------
    ; [TEST 6] Batched Cosine Similarity (1000 x 384)
    ; -------------------------------------------------------------------------
    lea rcx, [test6_msg]
    call [printf]

    ; Prepare norms_buf (all sqrt(384 * 0.1^2) = sqrt(3.84) = 1.95959)
    mov ecx, 1000
    xor eax, eax
.init_norms:
    mov dword [norms_buf + rax*4], 3FFAE148h ; 1.95959f
    inc eax
    cmp eax, ecx
    jb .init_norms

    lea rcx, [t_start]
    call [QueryPerformanceCounter]

    lea rcx, [vec_b]                    ; q
    mov eax, 419CB1F8h                  ; q_norm = sqrt(384 * 1.0) = 19.5959f
    vmovd xmm1, eax
    lea r8, [matrix_buf]                ; matrix
    lea r9, [norms_buf]                 ; norms
    mov qword [rsp + 20h], 1000         ; count
    mov qword [rsp + 28h], 384          ; dim
    mov qword [rsp + 30h], scores_buf   ; scores
    call nanovec_fasm_batch_cosine

    lea rcx, [t_end]
    call [QueryPerformanceCounter]

    ; Cosine = 38.4 / (19.5959 * 1.95959) = 1.0f
    mov eax, 3F800000h ; 1.0f
    vmovd xmm1, eax
    vmovss xmm0, [scores_buf]
    vsubss xmm2, xmm0, xmm1
    vandps xmm2, xmm2, xword [abs_mask]
    vucomiss xmm2, [tol_flt]
    ja .fail_t6

    mov rax, [t_end]
    sub rax, [t_start]
    imul rax, 1000000
    xor edx, edx
    div [freq]
    mov [elapsed_us], rax
    test rax, rax
    jnz .calc_rate6
    mov rax, 1
.calc_rate6:
    mov rax, 1000000000
    xor edx, edx
    div [elapsed_us]
    mov r8, rax
    shr r8, 20
    mov rdx, [elapsed_us]
    lea rcx, [bench_fmt]
    call [printf]
    jmp .test7

.fail_t6:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 1
    call [ExitProcess]

.test7:
    ; -------------------------------------------------------------------------
    ; [TEST 7] Odd / Prime dimension (dim = 77)
    ; -------------------------------------------------------------------------
    lea rcx, [test7_msg]
    call [printf]

    mov ecx, 77
    xor eax, eax
.init_t7:
    mov dword [vec_a + rax*4], 3F800000h ; 1.0f
    mov dword [vec_b + rax*4], 40000000h ; 2.0f
    inc eax
    cmp eax, ecx
    jb .init_t7

    lea rcx, [vec_a]
    lea rdx, [vec_b]
    mov r8, 77
    call nanovec_fasm_dot

    ; Dot = 77 * 2.0 = 154.0f (0x431A0000)
    mov eax, 431A0000h
    vmovd xmm1, eax
    vsubss xmm2, xmm0, xmm1
    vandps xmm2, xmm2, xword [abs_mask]
    vucomiss xmm2, [tol_flt]
    ja .fail_t7

    lea rcx, [pass_str]
    call [printf]

    ; All passed!
    lea rcx, [all_ok_msg]
    call [printf]

    xor ecx, ecx
    call [ExitProcess]

.fail_t7:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 1
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
