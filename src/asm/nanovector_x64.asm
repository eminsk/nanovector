; ==============================================================================
; NanoVector x64 AVX2/FMA Assembly Kernel
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; Windows x64 ABI Compliant (Volatile: RAX, RCX, RDX, R8-R11, XMM0-XMM5)
; Formatted for Flat Assembler (FASM) - 64-bit MS COFF
; ==============================================================================

format MS64 COFF

section '.text' code readable executable align 16

public nanovec_fasm_dot
public nanovec_fasm_l2_sq
public nanovec_fasm_batch_dot

; ------------------------------------------------------------------------------
; float nanovec_fasm_dot(const float* a, const float* b, size_t dim)
; RCX = a
; RDX = b
; R8  = dim
; Returns float in XMM0
; Uses only volatile registers: RAX, R10, YMM0-YMM5
; ------------------------------------------------------------------------------
align 16
nanovec_fasm_dot:
    vxorps ymm0, ymm0, ymm0     ; acc0
    vxorps ymm1, ymm1, ymm1     ; acc1

    test r8, r8
    jz .dot_done

    xor rax, rax                ; i = 0

.dot_loop16:
    lea r10, [rax + 16]
    cmp r10, r8
    ja .dot_tail8

    vmovups ymm2, [rcx + rax*4]
    vfmadd231ps ymm0, ymm2, [rdx + rax*4]

    vmovups ymm3, [rcx + rax*4 + 32]
    vfmadd231ps ymm1, ymm3, [rdx + rax*4 + 32]

    add rax, 16
    jmp .dot_loop16

.dot_tail8:
    vaddps ymm0, ymm0, ymm1

.dot_loop8:
    lea r10, [rax + 8]
    cmp r10, r8
    ja .dot_reduce

    vmovups ymm2, [rcx + rax*4]
    vfmadd231ps ymm0, ymm2, [rdx + rax*4]
    add rax, 8
    jmp .dot_loop8

.dot_reduce:
    ; Horizontal reduction of ymm0 to xmm0 (lower float)
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 04Eh ; swap 64-bit halves
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 001h ; move float 1 to 0
    vaddss xmm0, xmm0, xmm1

.dot_scalar:
    cmp rax, r8
    jae .dot_finish

    vmovss xmm2, [rcx + rax*4]
    vmulss xmm2, xmm2, [rdx + rax*4]
    vaddss xmm0, xmm0, xmm2
    inc rax
    jmp .dot_scalar

.dot_done:
    vzeroupper
    ret

.dot_finish:
    vzeroupper
    ret


; ------------------------------------------------------------------------------
; float nanovec_fasm_l2_sq(const float* a, const float* b, size_t dim)
; RCX = a
; RDX = b
; R8  = dim
; Returns float in XMM0
; Uses only volatile registers: RAX, R10, YMM0-YMM5
; ------------------------------------------------------------------------------
align 16
nanovec_fasm_l2_sq:
    vxorps ymm0, ymm0, ymm0     ; acc0
    vxorps ymm1, ymm1, ymm1     ; acc1

    test r8, r8
    jz .l2_done

    xor rax, rax                ; i = 0

.l2_loop16:
    lea r10, [rax + 16]
    cmp r10, r8
    ja .l2_tail8

    vmovups ymm2, [rcx + rax*4]
    vsubps  ymm2, ymm2, [rdx + rax*4]
    vfmadd231ps ymm0, ymm2, ymm2

    vmovups ymm3, [rcx + rax*4 + 32]
    vsubps  ymm3, ymm3, [rdx + rax*4 + 32]
    vfmadd231ps ymm1, ymm3, ymm3

    add rax, 16
    jmp .l2_loop16

.l2_tail8:
    vaddps ymm0, ymm0, ymm1

.l2_loop8:
    lea r10, [rax + 8]
    cmp r10, r8
    ja .l2_reduce

    vmovups ymm2, [rcx + rax*4]
    vsubps  ymm2, ymm2, [rdx + rax*4]
    vfmadd231ps ymm0, ymm2, ymm2
    add rax, 8
    jmp .l2_loop8

.l2_reduce:
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 04Eh
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 001h
    vaddss xmm0, xmm0, xmm1

.l2_scalar:
    cmp rax, r8
    jae .l2_finish

    vmovss xmm2, [rcx + rax*4]
    vsubss xmm2, xmm2, [rdx + rax*4]
    vmulss xmm2, xmm2, xmm2
    vaddss xmm0, xmm0, xmm2
    inc rax
    jmp .l2_scalar

.l2_done:
    vzeroupper
    ret

.l2_finish:
    vzeroupper
    ret


; ------------------------------------------------------------------------------
; void nanovec_fasm_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores)
; RCX = q
; RDX = matrix
; R8  = count
; R9  = dim
; [rsp + 40] = scores
; ------------------------------------------------------------------------------
align 16
nanovec_fasm_batch_dot:
    push rbp
    mov rbp, rsp
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 48                 ; 32 bytes shadow space + 16 bytes alignment

    mov r12, rcx                ; r12 = q
    mov r13, rdx                ; r13 = matrix
    mov r14, r8                 ; r14 = count
    mov rbx, r9                 ; rbx = dim
    mov rdi, [rbp + 48]         ; rdi = scores

    test r14, r14
    jz .batch_exit

    xor rsi, rsi                ; rsi = row

.batch_loop:
    cmp rsi, r14
    jae .batch_exit

    mov rcx, r12                ; arg1 = q
    mov rax, rsi
    imul rax, rbx
    shl rax, 2
    lea rdx, [r13 + rax]        ; arg2 = matrix + row*dim
    mov r8, rbx                 ; arg3 = dim

    call nanovec_fasm_dot

    vmovss [rdi + rsi*4], xmm0

    inc rsi
    jmp .batch_loop

.batch_exit:
    add rsp, 48
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    ret
