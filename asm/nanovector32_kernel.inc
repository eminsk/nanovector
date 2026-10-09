; =============================================================================
; NanoVector — 32-bit SSE2 Microkernels (x86 FASM)
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; MIT License
; Standard 32-bit cdecl Calling Convention (Float returned in ST0)
; =============================================================================

; -----------------------------------------------------------------------------
; float nanovec_fasm_dot(const float* a, const float* b, size_t dim)
; [ebp + 8]  = a
; [ebp + 12] = b
; [ebp + 16] = dim
; Returns float in ST(0) and XMM0
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_dot:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi
    sub esp, 16                 ; local temp for return float

    mov esi, [ebp + 8]          ; a
    mov edi, [ebp + 12]         ; b
    mov ecx, [ebp + 16]         ; dim

    xorps xmm0, xmm0
    xorps xmm1, xmm1
    xorps xmm2, xmm2
    xorps xmm3, xmm3

    test ecx, ecx
    jz .dot_done

    xor eax, eax                ; i = 0

.dot_loop16:
    lea edx, [eax + 16]
    cmp edx, ecx
    ja .dot_tail4

    movups xmm4, [esi + eax*4]
    movups xmm5, [edi + eax*4]
    mulps  xmm4, xmm5
    addps  xmm0, xmm4

    movups xmm4, [esi + eax*4 + 16]
    movups xmm5, [edi + eax*4 + 16]
    mulps  xmm4, xmm5
    addps  xmm1, xmm4

    movups xmm4, [esi + eax*4 + 32]
    movups xmm5, [edi + eax*4 + 32]
    mulps  xmm4, xmm5
    addps  xmm2, xmm4

    movups xmm4, [esi + eax*4 + 48]
    movups xmm5, [edi + eax*4 + 48]
    mulps  xmm4, xmm5
    addps  xmm3, xmm4

    add eax, 16
    jmp .dot_loop16

.dot_tail4:
    addps xmm0, xmm1
    addps xmm2, xmm3
    addps xmm0, xmm2

.dot_loop4:
    lea edx, [eax + 4]
    cmp edx, ecx
    ja .dot_reduce

    movups xmm4, [esi + eax*4]
    movups xmm5, [edi + eax*4]
    mulps  xmm4, xmm5
    addps  xmm0, xmm4
    add eax, 4
    jmp .dot_loop4

.dot_reduce:
    ; Horizontal reduction of xmm0
    movhlps xmm1, xmm0
    addps   xmm0, xmm1
    movaps  xmm1, xmm0
    shufps  xmm1, xmm1, 001h
    addss   xmm0, xmm1

.dot_scalar:
    cmp eax, ecx
    jae .dot_done

    movss xmm4, [esi + eax*4]
    mulss xmm4, [edi + eax*4]
    addss xmm0, xmm4
    inc eax
    jmp .dot_scalar

.dot_done:
    ; Return float in ST(0)
    movss [esp], xmm0
    fld dword [esp]

    add esp, 16
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret


; -----------------------------------------------------------------------------
; float nanovec_fasm_norm_sq(const float* a, size_t dim)
; [ebp + 8]  = a
; [ebp + 12] = dim
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_norm_sq:
    push ebp
    mov ebp, esp
    push dword [ebp + 12]       ; dim
    push dword [ebp + 8]        ; b = a
    push dword [ebp + 8]        ; a
    call nanovec_fasm_dot
    add esp, 12
    pop ebp
    ret


; -----------------------------------------------------------------------------
; float nanovec_fasm_l2_sq(const float* a, const float* b, size_t dim)
; [ebp + 8]  = a
; [ebp + 12] = b
; [ebp + 16] = dim
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_l2_sq:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi
    sub esp, 16

    mov esi, [ebp + 8]          ; a
    mov edi, [ebp + 12]         ; b
    mov ecx, [ebp + 16]         ; dim

    xorps xmm0, xmm0
    xorps xmm1, xmm1
    xorps xmm2, xmm2
    xorps xmm3, xmm3

    test ecx, ecx
    jz .l2_done

    xor eax, eax

.l2_loop16:
    lea edx, [eax + 16]
    cmp edx, ecx
    ja .l2_tail4

    movups xmm4, [esi + eax*4]
    subps  xmm4, [edi + eax*4]
    mulps  xmm4, xmm4
    addps  xmm0, xmm4

    movups xmm4, [esi + eax*4 + 16]
    subps  xmm4, [edi + eax*4 + 16]
    mulps  xmm4, xmm4
    addps  xmm1, xmm4

    movups xmm4, [esi + eax*4 + 32]
    subps  xmm4, [edi + eax*4 + 32]
    mulps  xmm4, xmm4
    addps  xmm2, xmm4

    movups xmm4, [esi + eax*4 + 48]
    subps  xmm4, [edi + eax*4 + 48]
    mulps  xmm4, xmm4
    addps  xmm3, xmm4

    add eax, 16
    jmp .l2_loop16

.l2_tail4:
    addps xmm0, xmm1
    addps xmm2, xmm3
    addps xmm0, xmm2

.l2_loop4:
    lea edx, [eax + 4]
    cmp edx, ecx
    ja .l2_reduce

    movups xmm4, [esi + eax*4]
    subps  xmm4, [edi + eax*4]
    mulps  xmm4, xmm4
    addps  xmm0, xmm4
    add eax, 4
    jmp .l2_loop4

.l2_reduce:
    movhlps xmm1, xmm0
    addps   xmm0, xmm1
    movaps  xmm1, xmm0
    shufps  xmm1, xmm1, 001h
    addss   xmm0, xmm1

.l2_scalar:
    cmp eax, ecx
    jae .l2_done

    movss xmm4, [esi + eax*4]
    subss xmm4, [edi + eax*4]
    mulss xmm4, xmm4
    addss xmm0, xmm4
    inc eax
    jmp .l2_scalar

.l2_done:
    movss [esp], xmm0
    fld dword [esp]

    add esp, 16
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret


; -----------------------------------------------------------------------------
; void nanovec_fasm_normalize(float* a, size_t dim)
; [ebp + 8]  = a
; [ebp + 12] = dim
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_normalize:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi
    sub esp, 16

    mov esi, [ebp + 8]          ; a
    mov ecx, [ebp + 12]         ; dim
    test ecx, ecx
    jz .norm_exit

    push ecx
    push esi
    call nanovec_fasm_norm_sq
    add esp, 8

    ; Result in ST(0)
    fstp dword [esp]
    movss xmm0, [esp]

    ; Check if norm_sq <= 1e-12f
    mov eax, 3369848695
    mov [esp], eax
    movss xmm1, [esp]
    ucomiss xmm0, xmm1
    jbe .norm_exit

    ; Compute 1.0f / sqrt(norm_sq)
    sqrtss xmm0, xmm0
    mov eax, 1065353216 ; 1.0f
    mov [esp], eax
    movss xmm1, [esp]
    divss xmm1, xmm0    ; xmm1 = inv_norm

    shufps xmm1, xmm1, 0 ; broadcast inv_norm to all 4 lanes

    xor eax, eax
.norm_loop4:
    lea edx, [eax + 4]
    cmp edx, ecx
    ja .norm_scalar

    movups xmm0, [esi + eax*4]
    mulps  xmm0, xmm1
    movups [esi + eax*4], xmm0
    add eax, 4
    jmp .norm_loop4

.norm_scalar:
    cmp eax, ecx
    jae .norm_exit

    movss xmm0, [esi + eax*4]
    mulss xmm0, xmm1
    movss [esi + eax*4], xmm0
    inc eax
    jmp .norm_scalar

.norm_exit:
    add esp, 16
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret


; -----------------------------------------------------------------------------
; void nanovec_fasm_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores)
; [ebp + 8]  = q
; [ebp + 12] = matrix
; [ebp + 16] = count
; [ebp + 20] = dim
; [ebp + 24] = scores
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_batch_dot:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi
    sub esp, 16

    mov esi, [ebp + 8]          ; q
    mov edi, [ebp + 12]         ; matrix
    mov ebx, [ebp + 16]         ; count

    test ebx, ebx
    jz .bdot_done

    xor eax, eax                ; row = 0

.bdot_loop:
    cmp eax, ebx
    jae .bdot_done

    mov [esp], eax              ; save row

    ; Calculate row_ptr = matrix + row * dim * 4
    mov ecx, [ebp + 20]         ; reload dim
    imul eax, ecx
    shl eax, 2
    add eax, edi                ; eax = row_ptr

    push ecx                    ; arg3: dim
    push eax                    ; arg2: row_ptr
    push esi                    ; arg1: q
    call nanovec_fasm_dot
    add esp, 12

    ; Result in ST(0) and XMM0
    mov eax, [esp]              ; restore row
    mov edx, [ebp + 24]         ; reload scores pointer!
    fstp dword [edx + eax*4]

    inc eax
    jmp .bdot_loop

.bdot_done:
    add esp, 16
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret


; -----------------------------------------------------------------------------
; void nanovec_fasm_batch_l2_sq(const float* q, const float* matrix, size_t count, size_t dim, float* scores)
; [ebp + 8]  = q
; [ebp + 12] = matrix
; [ebp + 16] = count
; [ebp + 20] = dim
; [ebp + 24] = scores
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_batch_l2_sq:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi
    sub esp, 16

    mov esi, [ebp + 8]          ; q
    mov edi, [ebp + 12]         ; matrix
    mov ebx, [ebp + 16]         ; count

    test ebx, ebx
    jz .bl2_done

    xor eax, eax

.bl2_loop:
    cmp eax, ebx
    jae .bl2_done

    mov [esp], eax

    mov ecx, [ebp + 20]         ; reload dim
    imul eax, ecx
    shl eax, 2
    add eax, edi                ; eax = row_ptr

    push ecx                    ; arg3: dim
    push eax                    ; arg2: row_ptr
    push esi                    ; arg1: q
    call nanovec_fasm_l2_sq
    add esp, 12

    mov eax, [esp]
    mov edx, [ebp + 24]         ; reload scores pointer!
    fstp dword [edx + eax*4]

    inc eax
    jmp .bl2_loop

.bl2_done:
    add esp, 16
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret


; -----------------------------------------------------------------------------
; void nanovec_fasm_batch_cosine(const float* q, float q_norm, const float* matrix, const float* norms, size_t count, size_t dim, float* scores)
; [ebp + 8]  = q
; [ebp + 12] = q_norm
; [ebp + 16] = matrix
; [ebp + 20] = norms
; [ebp + 24] = count
; [ebp + 28] = dim
; [ebp + 32] = scores
; -----------------------------------------------------------------------------
align 16
nanovec_fasm_batch_cosine:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi
    sub esp, 32                 ; [esp]=row, [esp+4]=inv_q_norm

    mov esi, [ebp + 8]          ; q
    mov ebx, [ebp + 24]         ; count

    test ebx, ebx
    jz .bcos_done

    ; Check q_norm <= 1e-12f
    movss xmm0, [ebp + 12]
    mov eax, 3369848695
    mov [esp], eax
    ucomiss xmm0, [esp]
    ja .bcos_valid_q

    ; Zero all scores
    mov edx, [ebp + 32]
    xor eax, eax
.bcos_zero_all:
    cmp eax, ebx
    jae .bcos_done
    mov dword [edx + eax*4], 0
    inc eax
    jmp .bcos_zero_all

.bcos_valid_q:
    ; Compute inv_q_norm = 1.0f / q_norm
    mov eax, 1065353216
    mov [esp], eax
    movss xmm1, [esp]
    divss xmm1, xmm0
    movss [esp + 4], xmm1       ; [esp+4] = inv_q_norm

    xor eax, eax                ; row = 0

.bcos_loop:
    cmp eax, ebx
    jae .bcos_done

    mov [esp], eax              ; save row

    ; Calculate row_ptr = matrix + row * dim * 4
    mov ecx, [ebp + 28]         ; reload dim
    imul eax, ecx
    shl eax, 2
    add eax, [ebp + 16]
    mov edi, eax                ; edi = row_ptr

    ; 1. Dot product
    push ecx                    ; dim
    push edi                    ; row_ptr
    push esi                    ; q
    call nanovec_fasm_dot
    add esp, 12

    ; Result in ST(0) and XMM0 = dot
    ; 2. Determine row_norm
    mov eax, [ebp + 20]         ; norms
    test eax, eax
    jz .bcos_calc_norm

    ; Precomputed norms array
    mov edx, [esp]              ; row
    movss xmm1, [eax + edx*4]   ; xmm1 = row_norm
    jmp .bcos_norm_check

.bcos_calc_norm:
    ; Compute norm_sq(row_ptr, dim)
    push dword [ebp + 28]       ; dim
    push edi                    ; row_ptr
    call nanovec_fasm_norm_sq
    add esp, 8
    ; Result in ST(0)
    fstp dword [esp + 8]
    movss xmm1, [esp + 8]
    sqrtss xmm1, xmm1           ; xmm1 = row_norm

.bcos_norm_check:
    ; Empty ST(0) from earlier nanovec_fasm_dot
    fstp dword [esp + 8]
    movss xmm0, [esp + 8]       ; xmm0 = dot

    ; Check row_norm > 1e-12f
    mov eax, 3369848695
    mov [esp + 12], eax
    ucomiss xmm1, [esp + 12]
    jbe .bcos_row_zero

    ; score = (dot * inv_q_norm) / row_norm
    mulss xmm0, [esp + 4]       ; xmm0 = dot * inv_q_norm
    divss xmm0, xmm1            ; xmm0 = score
    mov edx, [ebp + 32]
    mov eax, [esp]
    movss [edx + eax*4], xmm0
    jmp .bcos_next

.bcos_row_zero:
    mov edx, [ebp + 32]
    mov eax, [esp]
    mov dword [edx + eax*4], 0

.bcos_next:
    mov eax, [esp]
    inc eax
    jmp .bcos_loop

.bcos_done:
    add esp, 32
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret
