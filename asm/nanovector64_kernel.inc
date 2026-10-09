; =============================================================================
; NanoVector — High-Performance AVX2+FMA Microkernels (x86-64 FASM)
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; MIT License
; Windows x64 ABI Compliant
; =============================================================================

; -----------------------------------------------------------------------------
; float nanovec_fasm_dot(const float* a, const float* b, size_t dim)
; RCX = a, RDX = b, R8 = dim
; Returns float in XMM0
; Volatile registers used: RAX, R10, YMM0-YMM5
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_dot:
    vxorps ymm0, ymm0, ymm0     ; acc0
    vxorps ymm1, ymm1, ymm1     ; acc1
    vxorps ymm2, ymm2, ymm2     ; acc2
    vxorps ymm3, ymm3, ymm3     ; acc3

    test r8, r8
    jz .dot_done

    xor rax, rax                ; i = 0

.dot_loop32:
    lea r10, [rax + 32]
    cmp r10, r8
    ja .dot_tail16

    vmovups ymm4, [rcx + rax*4]
    vfmadd231ps ymm0, ymm4, [rdx + rax*4]

    vmovups ymm5, [rcx + rax*4 + 32]
    vfmadd231ps ymm1, ymm5, [rdx + rax*4 + 32]

    vmovups ymm4, [rcx + rax*4 + 64]
    vfmadd231ps ymm2, ymm4, [rdx + rax*4 + 64]

    vmovups ymm5, [rcx + rax*4 + 96]
    vfmadd231ps ymm3, ymm5, [rdx + rax*4 + 96]

    add rax, 32
    jmp .dot_loop32

.dot_tail16:
    vaddps ymm0, ymm0, ymm1
    vaddps ymm2, ymm2, ymm3
    vaddps ymm0, ymm0, ymm2

.dot_loop8:
    lea r10, [rax + 8]
    cmp r10, r8
    ja .dot_reduce

    vmovups ymm4, [rcx + rax*4]
    vfmadd231ps ymm0, ymm4, [rdx + rax*4]
    add rax, 8
    jmp .dot_loop8

.dot_reduce:
    ; Horizontal reduction of YMM0 into scalar XMM0
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 04Eh ; swap 64-bit halves
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 001h ; move float 1 to 0
    vaddss xmm0, xmm0, xmm1

.dot_scalar:
    cmp rax, r8
    jae .dot_done

    vmovss xmm4, [rcx + rax*4]
    vmulss xmm4, xmm4, [rdx + rax*4]
    vaddss xmm0, xmm0, xmm4
    inc rax
    jmp .dot_scalar

.dot_done:
    vzeroupper
    ret


; -----------------------------------------------------------------------------
; float nanovec_fasm_norm_sq(const float* a, size_t dim)
; RCX = a, RDX = dim
; Returns float in XMM0
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_norm_sq:
    mov r8, rdx         ; r8 = dim
    mov rdx, rcx        ; rdx = b = a
    jmp nanovec_fasm_dot


; -----------------------------------------------------------------------------
; float nanovec_fasm_l2_sq(const float* a, const float* b, size_t dim)
; RCX = a, RDX = b, R8 = dim
; Returns float in XMM0
; Volatile registers used: RAX, R10, YMM0-YMM5
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_l2_sq:
    vxorps ymm0, ymm0, ymm0     ; acc0
    vxorps ymm1, ymm1, ymm1     ; acc1
    vxorps ymm2, ymm2, ymm2     ; acc2
    vxorps ymm3, ymm3, ymm3     ; acc3

    test r8, r8
    jz .l2_done

    xor rax, rax                ; i = 0

.l2_loop32:
    lea r10, [rax + 32]
    cmp r10, r8
    ja .l2_tail16

    vmovups ymm4, [rcx + rax*4]
    vsubps  ymm4, ymm4, [rdx + rax*4]
    vfmadd231ps ymm0, ymm4, ymm4

    vmovups ymm5, [rcx + rax*4 + 32]
    vsubps  ymm5, ymm5, [rdx + rax*4 + 32]
    vfmadd231ps ymm1, ymm5, ymm5

    vmovups ymm4, [rcx + rax*4 + 64]
    vsubps  ymm4, ymm4, [rdx + rax*4 + 64]
    vfmadd231ps ymm2, ymm4, ymm4

    vmovups ymm5, [rcx + rax*4 + 96]
    vsubps  ymm5, ymm5, [rdx + rax*4 + 96]
    vfmadd231ps ymm3, ymm5, ymm5

    add rax, 32
    jmp .l2_loop32

.l2_tail16:
    vaddps ymm0, ymm0, ymm1
    vaddps ymm2, ymm2, ymm3
    vaddps ymm0, ymm0, ymm2

.l2_loop8:
    lea r10, [rax + 8]
    cmp r10, r8
    ja .l2_reduce

    vmovups ymm4, [rcx + rax*4]
    vsubps  ymm4, ymm4, [rdx + rax*4]
    vfmadd231ps ymm0, ymm4, ymm4
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
    jae .l2_done

    vmovss xmm4, [rcx + rax*4]
    vsubss xmm4, xmm4, [rdx + rax*4]
    vmulss xmm4, xmm4, xmm4
    vaddss xmm0, xmm0, xmm4
    inc rax
    jmp .l2_scalar

.l2_done:
    vzeroupper
    ret


; -----------------------------------------------------------------------------
; void nanovec_fasm_normalize(float* a, size_t dim)
; RCX = a, RDX = dim
; Normalizes vector a in-place
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_normalize:
    test rdx, rdx
    jz .norm_done

    push rbx
    push rsi
    sub rsp, 40

    mov rbx, rcx        ; rbx = a
    mov rsi, rdx        ; rsi = dim

    ; Calculate norm_sq
    mov rcx, rbx
    mov rdx, rsi
    call nanovec_fasm_norm_sq

    ; xmm0 has norm_sq
    ; if norm_sq <= 1e-12f, do nothing
    mov eax, 3369848695 ; 1e-12f binary representation approx (0x2E6D6067)
    vmovd xmm1, eax
    vucomiss xmm0, xmm1
    jbe .norm_exit

    ; Compute 1.0f / sqrt(norm_sq)
    vsqrtss xmm0, xmm0, xmm0
    mov eax, 1065353216 ; 1.0f (0x3F800000)
    vmovd xmm1, eax
    vdivss xmm1, xmm1, xmm0 ; xmm1 = inv_norm

    vbroadcastss ymm2, xmm1 ; ymm2 = [inv_norm, inv_norm, ...]

    xor rax, rax
.norm_loop8:
    lea r10, [rax + 8]
    cmp r10, rsi
    ja .norm_scalar

    vmovups ymm0, [rbx + rax*4]
    vmulps  ymm0, ymm0, ymm2
    vmovups [rbx + rax*4], ymm0
    add rax, 8
    jmp .norm_loop8

.norm_scalar:
    cmp rax, rsi
    jae .norm_exit

    vmovss xmm0, [rbx + rax*4]
    vmulss xmm0, xmm0, xmm1
    vmovss [rbx + rax*4], xmm0
    inc rax
    jmp .norm_scalar

.norm_exit:
    vzeroupper
    add rsp, 40
    pop rsi
    pop rbx
.norm_done:
    ret


; -----------------------------------------------------------------------------
; void nanovec_fasm_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores)
; RCX = q
; RDX = matrix
; R8  = count
; R9  = dim
; [rsp + 40] = scores
; -----------------------------------------------------------------------------
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
    jz .bdot_exit

    xor rsi, rsi                ; rsi = row

.bdot_loop:
    cmp rsi, r14
    jae .bdot_exit

    mov rcx, r12                ; arg1 = q
    mov rax, rsi
    imul rax, rbx
    shl rax, 2
    lea rdx, [r13 + rax]        ; arg2 = matrix + row*dim
    mov r8, rbx                 ; arg3 = dim

    call nanovec_fasm_dot

    vmovss [rdi + rsi*4], xmm0

    inc rsi
    jmp .bdot_loop

.bdot_exit:
    add rsp, 48
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    ret


; -----------------------------------------------------------------------------
; void nanovec_fasm_batch_l2_sq(const float* q, const float* matrix, size_t count, size_t dim, float* scores)
; RCX = q
; RDX = matrix
; R8  = count
; R9  = dim
; [rsp + 40] = scores
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_batch_l2_sq:
    push rbp
    mov rbp, rsp
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 48

    mov r12, rcx                ; r12 = q
    mov r13, rdx                ; r13 = matrix
    mov r14, r8                 ; r14 = count
    mov rbx, r9                 ; rbx = dim
    mov rdi, [rbp + 48]         ; rdi = scores

    test r14, r14
    jz .bl2_exit

    xor rsi, rsi                ; rsi = row

.bl2_loop:
    cmp rsi, r14
    jae .bl2_exit

    mov rcx, r12                ; arg1 = q
    mov rax, rsi
    imul rax, rbx
    shl rax, 2
    lea rdx, [r13 + rax]        ; arg2 = matrix + row*dim
    mov r8, rbx                 ; arg3 = dim

    call nanovec_fasm_l2_sq

    vmovss [rdi + rsi*4], xmm0

    inc rsi
    jmp .bl2_loop

.bl2_exit:
    add rsp, 48
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    ret


; -----------------------------------------------------------------------------
; void nanovec_fasm_batch_cosine(const float* q, float q_norm, const float* matrix, const float* norms, size_t count, size_t dim, float* scores)
; RCX = q
; XMM1 = q_norm
; R8  = matrix
; R9  = norms (can be NULL)
; [RBP + 48] = count
; [RBP + 56] = dim
; [RBP + 64] = scores
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_batch_cosine:
    push rbp
    mov rbp, rsp
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64

    mov r12, rcx                ; r12 = q
    mov r13, r8                 ; r13 = matrix
    mov r14, r9                 ; r14 = norms
    mov r15, [rbp + 48]         ; r15 = count
    mov rbx, [rbp + 56]         ; rbx = dim
    mov rdi, [rbp + 64]         ; rdi = scores

    test r15, r15
    jz .bcos_exit

    ; Check q_norm <= 1e-12f
    mov eax, 3369848695         ; 1e-12f
    vmovd xmm2, eax
    vucomiss xmm1, xmm2
    ja .bcos_valid_q

    ; Zero all scores if q_norm is near zero
    xor rsi, rsi
    vxorps xmm0, xmm0, xmm0
.bcos_zero_loop:
    cmp rsi, r15
    jae .bcos_exit
    vmovss [rdi + rsi*4], xmm0
    inc rsi
    jmp .bcos_zero_loop

.bcos_valid_q:
    ; Compute 1.0f / q_norm and save in [rsp + 32]
    mov eax, 1065353216         ; 1.0f
    vmovd xmm0, eax
    vdivss xmm0, xmm0, xmm1     ; xmm0 = inv_q_norm
    vmovss [rsp + 32], xmm0

    xor rsi, rsi                ; rsi = row

.bcos_loop:
    cmp rsi, r15
    jae .bcos_exit

    ; row_ptr = matrix + rsi * dim * 4
    mov rax, rsi
    imul rax, rbx
    shl rax, 2
    lea rdx, [r13 + rax]        ; rdx = row_ptr

    ; 1. Compute dot product
    mov rcx, r12                ; rcx = q
    mov r8, rbx                 ; r8 = dim
    call nanovec_fasm_dot
    ; xmm0 = dot

    ; 2. Determine row_norm
    test r14, r14
    jz .bcos_calc_norm

    ; Load from precomputed norms array
    vmovss xmm1, [r14 + rsi*4]
    jmp .bcos_norm_ready

.bcos_calc_norm:
    ; Save dot on stack
    vmovss [rsp + 40], xmm0
    ; Calculate row norm
    mov rax, rsi
    imul rax, rbx
    shl rax, 2
    lea rcx, [r13 + rax]        ; rcx = row_ptr
    mov rdx, rbx                ; rdx = dim
    call nanovec_fasm_norm_sq
    vsqrtss xmm1, xmm1, xmm0    ; xmm1 = row_norm
    vmovss xmm0, [rsp + 40]     ; restore dot

.bcos_norm_ready:
    ; Check row_norm > 1e-12f
    mov eax, 3369848695
    vmovd xmm3, eax
    vucomiss xmm1, xmm3
    jbe .bcos_row_zero

    ; score = (dot * inv_q_norm) / row_norm
    vmulss xmm0, xmm0, [rsp + 32]
    vdivss xmm0, xmm0, xmm1
    vmovss [rdi + rsi*4], xmm0
    jmp .bcos_next

.bcos_row_zero:
    vxorps xmm0, xmm0, xmm0
    vmovss [rdi + rsi*4], xmm0

.bcos_next:
    inc rsi
    jmp .bcos_loop

.bcos_exit:
    vzeroupper
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    ret
