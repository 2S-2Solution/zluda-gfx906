; zluda_aten_float_vec4.hsaco - disassembly (llvm-objdump 17, ROCm 5.7)
; sha256 4cb4278373743adfb6ec8509a973026933a7c4bcfad4240e7f1ecec8e2673ff8, 12456 bytes, gfx906
; The original .hip source was not kept; this listing is the reviewable form of the blob.
; See the comment above SUPPLEMENT_HSACO in module.rs.


zluda_aten_float_vec4.hsaco:	file format elf64-amdgpu

SYMBOL TABLE:
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_pow_float.private_seg_size
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_pow_float.num_agpr
0000000000000006 l       *ABS*	0000000000000000 zluda_vec4_pow_float.num_vgpr
000000000000000a l       *ABS*	0000000000000000 zluda_vec4_pow_float.numbered_sgpr
0000000000000001 l       *ABS*	0000000000000000 zluda_vec4_pow_float.uses_vcc
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_pow_float.uses_flat_scratch
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_pow_float.has_dyn_sized_stack
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_pow_float.has_recursion
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_sqrt_float.private_seg_size
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_sqrt_float.num_agpr
000000000000000b l       *ABS*	0000000000000000 zluda_vec4_sqrt_float.num_vgpr
000000000000000c l       *ABS*	0000000000000000 zluda_vec4_sqrt_float.numbered_sgpr
0000000000000001 l       *ABS*	0000000000000000 zluda_vec4_sqrt_float.uses_vcc
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_sqrt_float.uses_flat_scratch
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_sqrt_float.has_dyn_sized_stack
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_sqrt_float.has_recursion
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_mul_float.private_seg_size
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_mul_float.num_agpr
0000000000000006 l       *ABS*	0000000000000000 zluda_vec4_mul_float.num_vgpr
0000000000000009 l       *ABS*	0000000000000000 zluda_vec4_mul_float.numbered_sgpr
0000000000000001 l       *ABS*	0000000000000000 zluda_vec4_mul_float.uses_vcc
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_mul_float.uses_flat_scratch
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_mul_float.has_dyn_sized_stack
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_mul_float.has_recursion
0000000000000000 l       *ABS*	0000000000000000 zluda_unrolled_copy_float.private_seg_size
0000000000000000 l       *ABS*	0000000000000000 zluda_unrolled_copy_float.num_agpr
000000000000000b l       *ABS*	0000000000000000 zluda_unrolled_copy_float.num_vgpr
0000000000000011 l       *ABS*	0000000000000000 zluda_unrolled_copy_float.numbered_sgpr
0000000000000001 l       *ABS*	0000000000000000 zluda_unrolled_copy_float.uses_vcc
0000000000000000 l       *ABS*	0000000000000000 zluda_unrolled_copy_float.uses_flat_scratch
0000000000000000 l       *ABS*	0000000000000000 zluda_unrolled_copy_float.has_dyn_sized_stack
0000000000000000 l       *ABS*	0000000000000000 zluda_unrolled_copy_float.has_recursion
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_sq_float.private_seg_size
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_sq_float.num_agpr
0000000000000006 l       *ABS*	0000000000000000 zluda_vec4_sq_float.num_vgpr
0000000000000008 l       *ABS*	0000000000000000 zluda_vec4_sq_float.numbered_sgpr
0000000000000001 l       *ABS*	0000000000000000 zluda_vec4_sq_float.uses_vcc
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_sq_float.uses_flat_scratch
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_sq_float.has_dyn_sized_stack
0000000000000000 l       *ABS*	0000000000000000 zluda_vec4_sq_float.has_recursion
0000000000000000  w    F .text	0000000000000008 __cxa_pure_virtual
0000000000000008  w    F .text	0000000000000008 __cxa_deleted_virtual
0000000000000010  w    F .text	0000000000000008 .hidden _wassert
0000000000000100 g     F .text	0000000000000170 .protected zluda_vec4_pow_float
0000000000000000 g     O .rodata	0000000000000040 .protected zluda_vec4_pow_float.kd
0000000000000300 g     F .text	0000000000000320 .protected zluda_vec4_sqrt_float
0000000000000040 g     O .rodata	0000000000000040 .protected zluda_vec4_sqrt_float.kd
0000000000000700 g     F .text	0000000000000158 .protected zluda_vec4_mul_float
0000000000000080 g     O .rodata	0000000000000040 .protected zluda_vec4_mul_float.kd
0000000000000900 g     F .text	0000000000000a00 .protected zluda_unrolled_copy_float
00000000000000c0 g     O .rodata	0000000000000040 .protected zluda_unrolled_copy_float.kd
0000000000001300 g     F .text	0000000000000150 .protected zluda_vec4_sq_float
0000000000000100 g     O .rodata	0000000000000040 .protected zluda_vec4_sq_float.kd
0000000000000000 g     O .bss	0000000000000001 __hip_cuid_105e97f3003439ca

Disassembly of section .text:

0000000000000000 <__cxa_pure_virtual>:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 000000000000: BF8C0000
	s_trap 2                                                   // 000000000004: BF920002

0000000000000008 <__cxa_deleted_virtual>:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 000000000008: BF8C0000
	s_trap 2                                                   // 00000000000C: BF920002

0000000000000010 <_wassert>:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 000000000010: BF8C0000
	s_trap 2                                                   // 000000000014: BF920002
	s_nop 0                                                    // 000000000018: BF800000
	s_nop 0                                                    // 00000000001C: BF800000
	s_nop 0                                                    // 000000000020: BF800000
	s_nop 0                                                    // 000000000024: BF800000
	s_nop 0                                                    // 000000000028: BF800000
	s_nop 0                                                    // 00000000002C: BF800000
	s_nop 0                                                    // 000000000030: BF800000
	s_nop 0                                                    // 000000000034: BF800000
	s_nop 0                                                    // 000000000038: BF800000
	s_nop 0                                                    // 00000000003C: BF800000
	s_nop 0                                                    // 000000000040: BF800000
	s_nop 0                                                    // 000000000044: BF800000
	s_nop 0                                                    // 000000000048: BF800000
	s_nop 0                                                    // 00000000004C: BF800000
	s_nop 0                                                    // 000000000050: BF800000
	s_nop 0                                                    // 000000000054: BF800000
	s_nop 0                                                    // 000000000058: BF800000
	s_nop 0                                                    // 00000000005C: BF800000
	s_nop 0                                                    // 000000000060: BF800000
	s_nop 0                                                    // 000000000064: BF800000
	s_nop 0                                                    // 000000000068: BF800000
	s_nop 0                                                    // 00000000006C: BF800000
	s_nop 0                                                    // 000000000070: BF800000
	s_nop 0                                                    // 000000000074: BF800000
	s_nop 0                                                    // 000000000078: BF800000
	s_nop 0                                                    // 00000000007C: BF800000
	s_nop 0                                                    // 000000000080: BF800000
	s_nop 0                                                    // 000000000084: BF800000
	s_nop 0                                                    // 000000000088: BF800000
	s_nop 0                                                    // 00000000008C: BF800000
	s_nop 0                                                    // 000000000090: BF800000
	s_nop 0                                                    // 000000000094: BF800000
	s_nop 0                                                    // 000000000098: BF800000
	s_nop 0                                                    // 00000000009C: BF800000
	s_nop 0                                                    // 0000000000A0: BF800000
	s_nop 0                                                    // 0000000000A4: BF800000
	s_nop 0                                                    // 0000000000A8: BF800000
	s_nop 0                                                    // 0000000000AC: BF800000
	s_nop 0                                                    // 0000000000B0: BF800000
	s_nop 0                                                    // 0000000000B4: BF800000
	s_nop 0                                                    // 0000000000B8: BF800000
	s_nop 0                                                    // 0000000000BC: BF800000
	s_nop 0                                                    // 0000000000C0: BF800000
	s_nop 0                                                    // 0000000000C4: BF800000
	s_nop 0                                                    // 0000000000C8: BF800000
	s_nop 0                                                    // 0000000000CC: BF800000
	s_nop 0                                                    // 0000000000D0: BF800000
	s_nop 0                                                    // 0000000000D4: BF800000
	s_nop 0                                                    // 0000000000D8: BF800000
	s_nop 0                                                    // 0000000000DC: BF800000
	s_nop 0                                                    // 0000000000E0: BF800000
	s_nop 0                                                    // 0000000000E4: BF800000
	s_nop 0                                                    // 0000000000E8: BF800000
	s_nop 0                                                    // 0000000000EC: BF800000
	s_nop 0                                                    // 0000000000F0: BF800000
	s_nop 0                                                    // 0000000000F4: BF800000
	s_nop 0                                                    // 0000000000F8: BF800000
	s_nop 0                                                    // 0000000000FC: BF800000

0000000000000100 <zluda_vec4_pow_float>:
	s_load_dwordx2 s[8:9], s[4:5], 0x0                         // 000000000100: C0060202 00000000
	s_load_dwordx4 s[0:3], s[4:5], 0x8                         // 000000000108: C00A0002 00000008
	v_lshlrev_b32_e32 v0, 2, v0                                // 000000000110: 24000082
	v_lshl_or_b32 v0, s6, 9, v0                                // 000000000114: D2000000 04011206
	v_ashrrev_i32_e32 v1, 31, v0                               // 00000000011C: 2202009F
	s_waitcnt lgkmcnt(0)                                       // 000000000120: BF8CC07F
	v_cmp_gt_i32_e32 vcc, s8, v0                               // 000000000124: 7D880008
	s_and_saveexec_b64 s[4:5], vcc                             // 000000000128: BE84206A
	s_cbranch_execz 16                                         // 00000000012C: BF880010 <zluda_vec4_pow_float+0x70>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 000000000130: D28F0002 00020082
	v_mov_b32_e32 v5, s3                                       // 000000000138: 7E0A0203
	v_add_co_u32_e32 v4, vcc, s2, v2                           // 00000000013C: 32080402
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 000000000140: 380A0705
	global_load_dword v4, v[4:5], off                          // 000000000144: DC508000 047F0004
	v_mov_b32_e32 v5, s1                                       // 00000000014C: 7E0A0201
	v_add_co_u32_e32 v2, vcc, s0, v2                           // 000000000150: 32040400
	v_addc_co_u32_e32 v3, vcc, v5, v3, vcc                     // 000000000154: 38060705
	s_waitcnt vmcnt(0)                                         // 000000000158: BF8C0F70
	v_log_f32_e32 v4, v4                                       // 00000000015C: 7E084304
	v_mul_legacy_f32_e32 v4, s9, v4                            // 000000000160: 08080809
	v_exp_f32_e32 v4, v4                                       // 000000000164: 7E084104
	global_store_dword v[2:3], v4, off                         // 000000000168: DC708000 007F0402
	s_or_b64 exec, exec, s[4:5]                                // 000000000170: 87FE047E
	v_or_b32_e32 v2, 1, v0                                     // 000000000174: 28040081
	v_cmp_gt_i32_e32 vcc, s8, v2                               // 000000000178: 7D880408
	s_and_saveexec_b64 s[4:5], vcc                             // 00000000017C: BE84206A
	s_cbranch_execz 16                                         // 000000000180: BF880010 <zluda_vec4_pow_float+0xc4>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 000000000184: D28F0002 00020082
	v_mov_b32_e32 v5, s3                                       // 00000000018C: 7E0A0203
	v_add_co_u32_e32 v4, vcc, s2, v2                           // 000000000190: 32080402
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 000000000194: 380A0705
	global_load_dword v4, v[4:5], off offset:4                 // 000000000198: DC508004 047F0004
	v_mov_b32_e32 v5, s1                                       // 0000000001A0: 7E0A0201
	v_add_co_u32_e32 v2, vcc, s0, v2                           // 0000000001A4: 32040400
	v_addc_co_u32_e32 v3, vcc, v5, v3, vcc                     // 0000000001A8: 38060705
	s_waitcnt vmcnt(0)                                         // 0000000001AC: BF8C0F70
	v_log_f32_e32 v4, v4                                       // 0000000001B0: 7E084304
	v_mul_legacy_f32_e32 v4, s9, v4                            // 0000000001B4: 08080809
	v_exp_f32_e32 v4, v4                                       // 0000000001B8: 7E084104
	global_store_dword v[2:3], v4, off offset:4                // 0000000001BC: DC708004 007F0402
	s_or_b64 exec, exec, s[4:5]                                // 0000000001C4: 87FE047E
	v_or_b32_e32 v2, 2, v0                                     // 0000000001C8: 28040082
	v_cmp_gt_i32_e32 vcc, s8, v2                               // 0000000001CC: 7D880408
	s_and_saveexec_b64 s[4:5], vcc                             // 0000000001D0: BE84206A
	s_cbranch_execz 16                                         // 0000000001D4: BF880010 <zluda_vec4_pow_float+0x118>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 0000000001D8: D28F0002 00020082
	v_mov_b32_e32 v5, s3                                       // 0000000001E0: 7E0A0203
	v_add_co_u32_e32 v4, vcc, s2, v2                           // 0000000001E4: 32080402
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 0000000001E8: 380A0705
	global_load_dword v4, v[4:5], off offset:8                 // 0000000001EC: DC508008 047F0004
	v_mov_b32_e32 v5, s1                                       // 0000000001F4: 7E0A0201
	v_add_co_u32_e32 v2, vcc, s0, v2                           // 0000000001F8: 32040400
	v_addc_co_u32_e32 v3, vcc, v5, v3, vcc                     // 0000000001FC: 38060705
	s_waitcnt vmcnt(0)                                         // 000000000200: BF8C0F70
	v_log_f32_e32 v4, v4                                       // 000000000204: 7E084304
	v_mul_legacy_f32_e32 v4, s9, v4                            // 000000000208: 08080809
	v_exp_f32_e32 v4, v4                                       // 00000000020C: 7E084104
	global_store_dword v[2:3], v4, off offset:8                // 000000000210: DC708008 007F0402
	s_or_b64 exec, exec, s[4:5]                                // 000000000218: 87FE047E
	v_or_b32_e32 v2, 3, v0                                     // 00000000021C: 28040083
	v_cmp_gt_i32_e32 vcc, s8, v2                               // 000000000220: 7D880408
	s_and_saveexec_b64 s[4:5], vcc                             // 000000000224: BE84206A
	s_cbranch_execz 16                                         // 000000000228: BF880010 <zluda_vec4_pow_float+0x16c>
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 00000000022C: D28F0000 00020082
	v_mov_b32_e32 v3, s3                                       // 000000000234: 7E060203
	v_add_co_u32_e32 v2, vcc, s2, v0                           // 000000000238: 32040002
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc                     // 00000000023C: 38060303
	global_load_dword v2, v[2:3], off offset:12                // 000000000240: DC50800C 027F0002
	v_mov_b32_e32 v3, s1                                       // 000000000248: 7E060201
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 00000000024C: 32000000
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 000000000250: 38020303
	s_waitcnt vmcnt(0)                                         // 000000000254: BF8C0F70
	v_log_f32_e32 v2, v2                                       // 000000000258: 7E044302
	v_mul_legacy_f32_e32 v2, s9, v2                            // 00000000025C: 08040409
	v_exp_f32_e32 v2, v2                                       // 000000000260: 7E044102
	global_store_dword v[0:1], v2, off offset:12               // 000000000264: DC70800C 007F0200
	s_endpgm                                                   // 00000000026C: BF810000
	s_nop 0                                                    // 000000000270: BF800000
	s_nop 0                                                    // 000000000274: BF800000
	s_nop 0                                                    // 000000000278: BF800000
	s_nop 0                                                    // 00000000027C: BF800000
	s_nop 0                                                    // 000000000280: BF800000
	s_nop 0                                                    // 000000000284: BF800000
	s_nop 0                                                    // 000000000288: BF800000
	s_nop 0                                                    // 00000000028C: BF800000
	s_nop 0                                                    // 000000000290: BF800000
	s_nop 0                                                    // 000000000294: BF800000
	s_nop 0                                                    // 000000000298: BF800000
	s_nop 0                                                    // 00000000029C: BF800000
	s_nop 0                                                    // 0000000002A0: BF800000
	s_nop 0                                                    // 0000000002A4: BF800000
	s_nop 0                                                    // 0000000002A8: BF800000
	s_nop 0                                                    // 0000000002AC: BF800000
	s_nop 0                                                    // 0000000002B0: BF800000
	s_nop 0                                                    // 0000000002B4: BF800000
	s_nop 0                                                    // 0000000002B8: BF800000
	s_nop 0                                                    // 0000000002BC: BF800000
	s_nop 0                                                    // 0000000002C0: BF800000
	s_nop 0                                                    // 0000000002C4: BF800000
	s_nop 0                                                    // 0000000002C8: BF800000
	s_nop 0                                                    // 0000000002CC: BF800000
	s_nop 0                                                    // 0000000002D0: BF800000
	s_nop 0                                                    // 0000000002D4: BF800000
	s_nop 0                                                    // 0000000002D8: BF800000
	s_nop 0                                                    // 0000000002DC: BF800000
	s_nop 0                                                    // 0000000002E0: BF800000
	s_nop 0                                                    // 0000000002E4: BF800000
	s_nop 0                                                    // 0000000002E8: BF800000
	s_nop 0                                                    // 0000000002EC: BF800000
	s_nop 0                                                    // 0000000002F0: BF800000
	s_nop 0                                                    // 0000000002F4: BF800000
	s_nop 0                                                    // 0000000002F8: BF800000
	s_nop 0                                                    // 0000000002FC: BF800000

0000000000000300 <zluda_vec4_sqrt_float>:
	s_load_dword s7, s[4:5], 0x0                               // 000000000300: C00201C2 00000000
	s_load_dwordx4 s[8:11], s[4:5], 0x8                        // 000000000308: C00A0202 00000008
	v_lshlrev_b32_e32 v0, 2, v0                                // 000000000310: 24000082
	v_lshl_or_b32 v0, s6, 9, v0                                // 000000000314: D2000000 04011206
	v_ashrrev_i32_e32 v1, 31, v0                               // 00000000031C: 2202009F
	s_waitcnt lgkmcnt(0)                                       // 000000000320: BF8CC07F
	v_cmp_gt_i32_e32 vcc, s7, v0                               // 000000000324: 7D880007
	s_and_saveexec_b64 s[2:3], vcc                             // 000000000328: BE82206A
	s_cbranch_execz 43                                         // 00000000032C: BF88002B <zluda_vec4_sqrt_float+0xdc>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 000000000330: D28F0002 00020082
	v_mov_b32_e32 v5, s11                                      // 000000000338: 7E0A020B
	v_add_co_u32_e32 v4, vcc, s10, v2                          // 00000000033C: 3208040A
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 000000000340: 380A0705
	global_load_dword v4, v[4:5], off                          // 000000000344: DC508000 047F0004
	s_mov_b32 s0, 0xf800000                                    // 00000000034C: BE8000FF 0F800000
	v_mov_b32_e32 v5, s9                                       // 000000000354: 7E0A0209
	v_mov_b32_e32 v6, 0x260                                    // 000000000358: 7E0C02FF 00000260
	s_waitcnt vmcnt(0)                                         // 000000000360: BF8C0F70
	v_mul_f32_e32 v7, 0x4f800000, v4                           // 000000000364: 0A0E08FF 4F800000
	v_cmp_gt_f32_e32 vcc, s0, v4                               // 00000000036C: 7C880800
	v_cndmask_b32_e32 v4, v4, v7, vcc                          // 000000000370: 00080F04
	v_sqrt_f32_e32 v7, v4                                      // 000000000374: 7E0E4F04
	v_add_co_u32_e64 v2, s[0:1], s8, v2                        // 000000000378: D1190002 00020408
	v_addc_co_u32_e64 v3, s[0:1], v5, v3, s[0:1]               // 000000000380: D11C0003 00020705
	v_add_u32_e32 v5, -1, v7                                   // 000000000388: 680A0EC1
	v_add_u32_e32 v8, 1, v7                                    // 00000000038C: 68100E81
	v_fma_f32 v9, -v5, v7, v4                                  // 000000000390: D1CB0009 24120F05
	v_fma_f32 v10, -v8, v7, v4                                 // 000000000398: D1CB000A 24120F08
	v_cmp_ge_f32_e64 s[0:1], 0, v9                             // 0000000003A0: D0460000 00021280
	v_cndmask_b32_e64 v5, v7, v5, s[0:1]                       // 0000000003A8: D1000005 00020B07
	v_cmp_lt_f32_e64 s[0:1], 0, v10                            // 0000000003B0: D0410000 00021480
	v_cndmask_b32_e64 v5, v5, v8, s[0:1]                       // 0000000003B8: D1000005 00021105
	v_mul_f32_e32 v7, 0x37800000, v5                           // 0000000003C0: 0A0E0AFF 37800000
	v_cndmask_b32_e32 v5, v5, v7, vcc                          // 0000000003C8: 000A0F05
	v_cmp_class_f32_e32 vcc, v4, v6                            // 0000000003CC: 7C200D04
	v_cndmask_b32_e32 v4, v5, v4, vcc                          // 0000000003D0: 00080905
	global_store_dword v[2:3], v4, off                         // 0000000003D4: DC708000 007F0402
	s_or_b64 exec, exec, s[2:3]                                // 0000000003DC: 87FE027E
	v_or_b32_e32 v2, 1, v0                                     // 0000000003E0: 28040081
	v_cmp_gt_i32_e32 vcc, s7, v2                               // 0000000003E4: 7D880407
	s_and_saveexec_b64 s[2:3], vcc                             // 0000000003E8: BE82206A
	s_cbranch_execz 43                                         // 0000000003EC: BF88002B <zluda_vec4_sqrt_float+0x19c>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 0000000003F0: D28F0002 00020082
	v_mov_b32_e32 v5, s11                                      // 0000000003F8: 7E0A020B
	v_add_co_u32_e32 v4, vcc, s10, v2                          // 0000000003FC: 3208040A
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 000000000400: 380A0705
	global_load_dword v4, v[4:5], off offset:4                 // 000000000404: DC508004 047F0004
	s_mov_b32 s0, 0xf800000                                    // 00000000040C: BE8000FF 0F800000
	v_mov_b32_e32 v5, s9                                       // 000000000414: 7E0A0209
	v_mov_b32_e32 v6, 0x260                                    // 000000000418: 7E0C02FF 00000260
	s_waitcnt vmcnt(0)                                         // 000000000420: BF8C0F70
	v_mul_f32_e32 v7, 0x4f800000, v4                           // 000000000424: 0A0E08FF 4F800000
	v_cmp_gt_f32_e32 vcc, s0, v4                               // 00000000042C: 7C880800
	v_cndmask_b32_e32 v4, v4, v7, vcc                          // 000000000430: 00080F04
	v_sqrt_f32_e32 v7, v4                                      // 000000000434: 7E0E4F04
	v_add_co_u32_e64 v2, s[0:1], s8, v2                        // 000000000438: D1190002 00020408
	v_addc_co_u32_e64 v3, s[0:1], v5, v3, s[0:1]               // 000000000440: D11C0003 00020705
	v_add_u32_e32 v5, -1, v7                                   // 000000000448: 680A0EC1
	v_add_u32_e32 v8, 1, v7                                    // 00000000044C: 68100E81
	v_fma_f32 v9, -v5, v7, v4                                  // 000000000450: D1CB0009 24120F05
	v_fma_f32 v10, -v8, v7, v4                                 // 000000000458: D1CB000A 24120F08
	v_cmp_ge_f32_e64 s[0:1], 0, v9                             // 000000000460: D0460000 00021280
	v_cndmask_b32_e64 v5, v7, v5, s[0:1]                       // 000000000468: D1000005 00020B07
	v_cmp_lt_f32_e64 s[0:1], 0, v10                            // 000000000470: D0410000 00021480
	v_cndmask_b32_e64 v5, v5, v8, s[0:1]                       // 000000000478: D1000005 00021105
	v_mul_f32_e32 v7, 0x37800000, v5                           // 000000000480: 0A0E0AFF 37800000
	v_cndmask_b32_e32 v5, v5, v7, vcc                          // 000000000488: 000A0F05
	v_cmp_class_f32_e32 vcc, v4, v6                            // 00000000048C: 7C200D04
	v_cndmask_b32_e32 v4, v5, v4, vcc                          // 000000000490: 00080905
	global_store_dword v[2:3], v4, off offset:4                // 000000000494: DC708004 007F0402
	s_or_b64 exec, exec, s[2:3]                                // 00000000049C: 87FE027E
	v_or_b32_e32 v2, 2, v0                                     // 0000000004A0: 28040082
	v_cmp_gt_i32_e32 vcc, s7, v2                               // 0000000004A4: 7D880407
	s_and_saveexec_b64 s[2:3], vcc                             // 0000000004A8: BE82206A
	s_cbranch_execz 43                                         // 0000000004AC: BF88002B <zluda_vec4_sqrt_float+0x25c>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 0000000004B0: D28F0002 00020082
	v_mov_b32_e32 v5, s11                                      // 0000000004B8: 7E0A020B
	v_add_co_u32_e32 v4, vcc, s10, v2                          // 0000000004BC: 3208040A
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 0000000004C0: 380A0705
	global_load_dword v4, v[4:5], off offset:8                 // 0000000004C4: DC508008 047F0004
	s_mov_b32 s0, 0xf800000                                    // 0000000004CC: BE8000FF 0F800000
	v_mov_b32_e32 v5, s9                                       // 0000000004D4: 7E0A0209
	v_mov_b32_e32 v6, 0x260                                    // 0000000004D8: 7E0C02FF 00000260
	s_waitcnt vmcnt(0)                                         // 0000000004E0: BF8C0F70
	v_mul_f32_e32 v7, 0x4f800000, v4                           // 0000000004E4: 0A0E08FF 4F800000
	v_cmp_gt_f32_e32 vcc, s0, v4                               // 0000000004EC: 7C880800
	v_cndmask_b32_e32 v4, v4, v7, vcc                          // 0000000004F0: 00080F04
	v_sqrt_f32_e32 v7, v4                                      // 0000000004F4: 7E0E4F04
	v_add_co_u32_e64 v2, s[0:1], s8, v2                        // 0000000004F8: D1190002 00020408
	v_addc_co_u32_e64 v3, s[0:1], v5, v3, s[0:1]               // 000000000500: D11C0003 00020705
	v_add_u32_e32 v5, -1, v7                                   // 000000000508: 680A0EC1
	v_add_u32_e32 v8, 1, v7                                    // 00000000050C: 68100E81
	v_fma_f32 v9, -v5, v7, v4                                  // 000000000510: D1CB0009 24120F05
	v_fma_f32 v10, -v8, v7, v4                                 // 000000000518: D1CB000A 24120F08
	v_cmp_ge_f32_e64 s[0:1], 0, v9                             // 000000000520: D0460000 00021280
	v_cndmask_b32_e64 v5, v7, v5, s[0:1]                       // 000000000528: D1000005 00020B07
	v_cmp_lt_f32_e64 s[0:1], 0, v10                            // 000000000530: D0410000 00021480
	v_cndmask_b32_e64 v5, v5, v8, s[0:1]                       // 000000000538: D1000005 00021105
	v_mul_f32_e32 v7, 0x37800000, v5                           // 000000000540: 0A0E0AFF 37800000
	v_cndmask_b32_e32 v5, v5, v7, vcc                          // 000000000548: 000A0F05
	v_cmp_class_f32_e32 vcc, v4, v6                            // 00000000054C: 7C200D04
	v_cndmask_b32_e32 v4, v5, v4, vcc                          // 000000000550: 00080905
	global_store_dword v[2:3], v4, off offset:8                // 000000000554: DC708008 007F0402
	s_or_b64 exec, exec, s[2:3]                                // 00000000055C: 87FE027E
	v_or_b32_e32 v2, 3, v0                                     // 000000000560: 28040083
	v_cmp_gt_i32_e32 vcc, s7, v2                               // 000000000564: 7D880407
	s_and_saveexec_b64 s[0:1], vcc                             // 000000000568: BE80206A
	s_cbranch_execz 43                                         // 00000000056C: BF88002B <zluda_vec4_sqrt_float+0x31c>
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000000570: D28F0000 00020082
	v_mov_b32_e32 v3, s11                                      // 000000000578: 7E06020B
	v_add_co_u32_e32 v2, vcc, s10, v0                          // 00000000057C: 3204000A
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc                     // 000000000580: 38060303
	global_load_dword v2, v[2:3], off offset:12                // 000000000584: DC50800C 027F0002
	s_mov_b32 s0, 0xf800000                                    // 00000000058C: BE8000FF 0F800000
	v_mov_b32_e32 v3, s9                                       // 000000000594: 7E060209
	v_mov_b32_e32 v4, 0x260                                    // 000000000598: 7E0802FF 00000260
	s_waitcnt vmcnt(0)                                         // 0000000005A0: BF8C0F70
	v_mul_f32_e32 v5, 0x4f800000, v2                           // 0000000005A4: 0A0A04FF 4F800000
	v_cmp_gt_f32_e32 vcc, s0, v2                               // 0000000005AC: 7C880400
	v_cndmask_b32_e32 v2, v2, v5, vcc                          // 0000000005B0: 00040B02
	v_sqrt_f32_e32 v5, v2                                      // 0000000005B4: 7E0A4F02
	v_add_co_u32_e64 v0, s[0:1], s8, v0                        // 0000000005B8: D1190000 00020008
	v_addc_co_u32_e64 v1, s[0:1], v3, v1, s[0:1]               // 0000000005C0: D11C0001 00020303
	v_add_u32_e32 v3, -1, v5                                   // 0000000005C8: 68060AC1
	v_add_u32_e32 v6, 1, v5                                    // 0000000005CC: 680C0A81
	v_fma_f32 v7, -v3, v5, v2                                  // 0000000005D0: D1CB0007 240A0B03
	v_fma_f32 v8, -v6, v5, v2                                  // 0000000005D8: D1CB0008 240A0B06
	v_cmp_ge_f32_e64 s[0:1], 0, v7                             // 0000000005E0: D0460000 00020E80
	v_cndmask_b32_e64 v3, v5, v3, s[0:1]                       // 0000000005E8: D1000003 00020705
	v_cmp_lt_f32_e64 s[0:1], 0, v8                             // 0000000005F0: D0410000 00021080
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]                       // 0000000005F8: D1000003 00020D03
	v_mul_f32_e32 v5, 0x37800000, v3                           // 000000000600: 0A0A06FF 37800000
	v_cndmask_b32_e32 v3, v3, v5, vcc                          // 000000000608: 00060B03
	v_cmp_class_f32_e32 vcc, v2, v4                            // 00000000060C: 7C200902
	v_cndmask_b32_e32 v2, v3, v2, vcc                          // 000000000610: 00040503
	global_store_dword v[0:1], v2, off offset:12               // 000000000614: DC70800C 007F0200
	s_endpgm                                                   // 00000000061C: BF810000
	s_nop 0                                                    // 000000000620: BF800000
	s_nop 0                                                    // 000000000624: BF800000
	s_nop 0                                                    // 000000000628: BF800000
	s_nop 0                                                    // 00000000062C: BF800000
	s_nop 0                                                    // 000000000630: BF800000
	s_nop 0                                                    // 000000000634: BF800000
	s_nop 0                                                    // 000000000638: BF800000
	s_nop 0                                                    // 00000000063C: BF800000
	s_nop 0                                                    // 000000000640: BF800000
	s_nop 0                                                    // 000000000644: BF800000
	s_nop 0                                                    // 000000000648: BF800000
	s_nop 0                                                    // 00000000064C: BF800000
	s_nop 0                                                    // 000000000650: BF800000
	s_nop 0                                                    // 000000000654: BF800000
	s_nop 0                                                    // 000000000658: BF800000
	s_nop 0                                                    // 00000000065C: BF800000
	s_nop 0                                                    // 000000000660: BF800000
	s_nop 0                                                    // 000000000664: BF800000
	s_nop 0                                                    // 000000000668: BF800000
	s_nop 0                                                    // 00000000066C: BF800000
	s_nop 0                                                    // 000000000670: BF800000
	s_nop 0                                                    // 000000000674: BF800000
	s_nop 0                                                    // 000000000678: BF800000
	s_nop 0                                                    // 00000000067C: BF800000
	s_nop 0                                                    // 000000000680: BF800000
	s_nop 0                                                    // 000000000684: BF800000
	s_nop 0                                                    // 000000000688: BF800000
	s_nop 0                                                    // 00000000068C: BF800000
	s_nop 0                                                    // 000000000690: BF800000
	s_nop 0                                                    // 000000000694: BF800000
	s_nop 0                                                    // 000000000698: BF800000
	s_nop 0                                                    // 00000000069C: BF800000
	s_nop 0                                                    // 0000000006A0: BF800000
	s_nop 0                                                    // 0000000006A4: BF800000
	s_nop 0                                                    // 0000000006A8: BF800000
	s_nop 0                                                    // 0000000006AC: BF800000
	s_nop 0                                                    // 0000000006B0: BF800000
	s_nop 0                                                    // 0000000006B4: BF800000
	s_nop 0                                                    // 0000000006B8: BF800000
	s_nop 0                                                    // 0000000006BC: BF800000
	s_nop 0                                                    // 0000000006C0: BF800000
	s_nop 0                                                    // 0000000006C4: BF800000
	s_nop 0                                                    // 0000000006C8: BF800000
	s_nop 0                                                    // 0000000006CC: BF800000
	s_nop 0                                                    // 0000000006D0: BF800000
	s_nop 0                                                    // 0000000006D4: BF800000
	s_nop 0                                                    // 0000000006D8: BF800000
	s_nop 0                                                    // 0000000006DC: BF800000
	s_nop 0                                                    // 0000000006E0: BF800000
	s_nop 0                                                    // 0000000006E4: BF800000
	s_nop 0                                                    // 0000000006E8: BF800000
	s_nop 0                                                    // 0000000006EC: BF800000
	s_nop 0                                                    // 0000000006F0: BF800000
	s_nop 0                                                    // 0000000006F4: BF800000
	s_nop 0                                                    // 0000000006F8: BF800000
	s_nop 0                                                    // 0000000006FC: BF800000

0000000000000700 <zluda_vec4_mul_float>:
	s_load_dword s8, s[4:5], 0x0                               // 000000000700: C0020202 00000000
	s_load_dword s7, s[4:5], 0x8                               // 000000000708: C00201C2 00000008
	s_load_dwordx4 s[0:3], s[4:5], 0x10                        // 000000000710: C00A0002 00000010
	v_lshlrev_b32_e32 v0, 2, v0                                // 000000000718: 24000082
	v_lshl_or_b32 v0, s6, 9, v0                                // 00000000071C: D2000000 04011206
	s_waitcnt lgkmcnt(0)                                       // 000000000724: BF8CC07F
	v_cmp_gt_i32_e32 vcc, s8, v0                               // 000000000728: 7D880008
	v_ashrrev_i32_e32 v1, 31, v0                               // 00000000072C: 2202009F
	s_and_saveexec_b64 s[4:5], vcc                             // 000000000730: BE84206A
	s_cbranch_execz 14                                         // 000000000734: BF88000E <zluda_vec4_mul_float+0x70>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 000000000738: D28F0002 00020082
	v_mov_b32_e32 v5, s3                                       // 000000000740: 7E0A0203
	v_add_co_u32_e32 v4, vcc, s2, v2                           // 000000000744: 32080402
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 000000000748: 380A0705
	global_load_dword v4, v[4:5], off                          // 00000000074C: DC508000 047F0004
	v_mov_b32_e32 v5, s1                                       // 000000000754: 7E0A0201
	v_add_co_u32_e32 v2, vcc, s0, v2                           // 000000000758: 32040400
	v_addc_co_u32_e32 v3, vcc, v5, v3, vcc                     // 00000000075C: 38060705
	s_waitcnt vmcnt(0)                                         // 000000000760: BF8C0F70
	v_mul_f32_e32 v4, s7, v4                                   // 000000000764: 0A080807
	global_store_dword v[2:3], v4, off                         // 000000000768: DC708000 007F0402
	s_or_b64 exec, exec, s[4:5]                                // 000000000770: 87FE047E
	v_or_b32_e32 v2, 1, v0                                     // 000000000774: 28040081
	v_cmp_gt_i32_e32 vcc, s8, v2                               // 000000000778: 7D880408
	s_and_saveexec_b64 s[4:5], vcc                             // 00000000077C: BE84206A
	s_cbranch_execz 14                                         // 000000000780: BF88000E <zluda_vec4_mul_float+0xbc>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 000000000784: D28F0002 00020082
	v_mov_b32_e32 v5, s3                                       // 00000000078C: 7E0A0203
	v_add_co_u32_e32 v4, vcc, s2, v2                           // 000000000790: 32080402
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 000000000794: 380A0705
	global_load_dword v4, v[4:5], off offset:4                 // 000000000798: DC508004 047F0004
	v_mov_b32_e32 v5, s1                                       // 0000000007A0: 7E0A0201
	v_add_co_u32_e32 v2, vcc, s0, v2                           // 0000000007A4: 32040400
	v_addc_co_u32_e32 v3, vcc, v5, v3, vcc                     // 0000000007A8: 38060705
	s_waitcnt vmcnt(0)                                         // 0000000007AC: BF8C0F70
	v_mul_f32_e32 v4, s7, v4                                   // 0000000007B0: 0A080807
	global_store_dword v[2:3], v4, off offset:4                // 0000000007B4: DC708004 007F0402
	s_or_b64 exec, exec, s[4:5]                                // 0000000007BC: 87FE047E
	v_or_b32_e32 v2, 2, v0                                     // 0000000007C0: 28040082
	v_cmp_gt_i32_e32 vcc, s8, v2                               // 0000000007C4: 7D880408
	s_and_saveexec_b64 s[4:5], vcc                             // 0000000007C8: BE84206A
	s_cbranch_execz 14                                         // 0000000007CC: BF88000E <zluda_vec4_mul_float+0x108>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 0000000007D0: D28F0002 00020082
	v_mov_b32_e32 v5, s3                                       // 0000000007D8: 7E0A0203
	v_add_co_u32_e32 v4, vcc, s2, v2                           // 0000000007DC: 32080402
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 0000000007E0: 380A0705
	global_load_dword v4, v[4:5], off offset:8                 // 0000000007E4: DC508008 047F0004
	v_mov_b32_e32 v5, s1                                       // 0000000007EC: 7E0A0201
	v_add_co_u32_e32 v2, vcc, s0, v2                           // 0000000007F0: 32040400
	v_addc_co_u32_e32 v3, vcc, v5, v3, vcc                     // 0000000007F4: 38060705
	s_waitcnt vmcnt(0)                                         // 0000000007F8: BF8C0F70
	v_mul_f32_e32 v4, s7, v4                                   // 0000000007FC: 0A080807
	global_store_dword v[2:3], v4, off offset:8                // 000000000800: DC708008 007F0402
	s_or_b64 exec, exec, s[4:5]                                // 000000000808: 87FE047E
	v_or_b32_e32 v2, 3, v0                                     // 00000000080C: 28040083
	v_cmp_gt_i32_e32 vcc, s8, v2                               // 000000000810: 7D880408
	s_and_saveexec_b64 s[4:5], vcc                             // 000000000814: BE84206A
	s_cbranch_execz 14                                         // 000000000818: BF88000E <zluda_vec4_mul_float+0x154>
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 00000000081C: D28F0000 00020082
	v_mov_b32_e32 v3, s3                                       // 000000000824: 7E060203
	v_add_co_u32_e32 v2, vcc, s2, v0                           // 000000000828: 32040002
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc                     // 00000000082C: 38060303
	global_load_dword v2, v[2:3], off offset:12                // 000000000830: DC50800C 027F0002
	v_mov_b32_e32 v3, s1                                       // 000000000838: 7E060201
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 00000000083C: 32000000
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 000000000840: 38020303
	s_waitcnt vmcnt(0)                                         // 000000000844: BF8C0F70
	v_mul_f32_e32 v2, s7, v2                                   // 000000000848: 0A040407
	global_store_dword v[0:1], v2, off offset:12               // 00000000084C: DC70800C 007F0200
	s_endpgm                                                   // 000000000854: BF810000
	s_nop 0                                                    // 000000000858: BF800000
	s_nop 0                                                    // 00000000085C: BF800000
	s_nop 0                                                    // 000000000860: BF800000
	s_nop 0                                                    // 000000000864: BF800000
	s_nop 0                                                    // 000000000868: BF800000
	s_nop 0                                                    // 00000000086C: BF800000
	s_nop 0                                                    // 000000000870: BF800000
	s_nop 0                                                    // 000000000874: BF800000
	s_nop 0                                                    // 000000000878: BF800000
	s_nop 0                                                    // 00000000087C: BF800000
	s_nop 0                                                    // 000000000880: BF800000
	s_nop 0                                                    // 000000000884: BF800000
	s_nop 0                                                    // 000000000888: BF800000
	s_nop 0                                                    // 00000000088C: BF800000
	s_nop 0                                                    // 000000000890: BF800000
	s_nop 0                                                    // 000000000894: BF800000
	s_nop 0                                                    // 000000000898: BF800000
	s_nop 0                                                    // 00000000089C: BF800000
	s_nop 0                                                    // 0000000008A0: BF800000
	s_nop 0                                                    // 0000000008A4: BF800000
	s_nop 0                                                    // 0000000008A8: BF800000
	s_nop 0                                                    // 0000000008AC: BF800000
	s_nop 0                                                    // 0000000008B0: BF800000
	s_nop 0                                                    // 0000000008B4: BF800000
	s_nop 0                                                    // 0000000008B8: BF800000
	s_nop 0                                                    // 0000000008BC: BF800000
	s_nop 0                                                    // 0000000008C0: BF800000
	s_nop 0                                                    // 0000000008C4: BF800000
	s_nop 0                                                    // 0000000008C8: BF800000
	s_nop 0                                                    // 0000000008CC: BF800000
	s_nop 0                                                    // 0000000008D0: BF800000
	s_nop 0                                                    // 0000000008D4: BF800000
	s_nop 0                                                    // 0000000008D8: BF800000
	s_nop 0                                                    // 0000000008DC: BF800000
	s_nop 0                                                    // 0000000008E0: BF800000
	s_nop 0                                                    // 0000000008E4: BF800000
	s_nop 0                                                    // 0000000008E8: BF800000
	s_nop 0                                                    // 0000000008EC: BF800000
	s_nop 0                                                    // 0000000008F0: BF800000
	s_nop 0                                                    // 0000000008F4: BF800000
	s_nop 0                                                    // 0000000008F8: BF800000
	s_nop 0                                                    // 0000000008FC: BF800000

0000000000000900 <zluda_unrolled_copy_float>:
	s_load_dword s14, s[4:5], 0x0                              // 000000000900: C0020382 00000000
	s_load_dwordx4 s[0:3], s[4:5], 0x8                         // 000000000908: C00A0002 00000008
	s_load_dword s13, s[4:5], 0x1c                             // 000000000910: C0020342 0000001C
	s_load_dword s12, s[4:5], 0x24                             // 000000000918: C0020302 00000024
	v_lshlrev_b32_e32 v0, 2, v0                                // 000000000920: 24000082
	v_lshl_or_b32 v0, s6, 9, v0                                // 000000000924: D2000000 04011206
	s_waitcnt lgkmcnt(0)                                       // 00000000092C: BF8CC07F
	v_cmp_gt_i32_e32 vcc, s14, v0                              // 000000000930: 7D88000E
	s_and_saveexec_b64 s[4:5], vcc                             // 000000000934: BE84206A
	s_cbranch_execz 85                                         // 000000000938: BF880055 <zluda_unrolled_copy_float+0x190>
	s_and_b32 s15, s13, 0xff                                   // 00000000093C: 860FFF0D 000000FF
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000944: 2202009F
	s_cmp_lt_i32 s15, 7                                        // 000000000948: BF04870F
	s_mov_b64 s[6:7], 0                                        // 00000000094C: BE860180
	s_cbranch_scc1 17                                          // 000000000950: BF850011 <zluda_unrolled_copy_float+0x98>
	s_and_b32 s16, 0xffff, s15                                 // 000000000954: 86100FFF 0000FFFF
	s_cmp_gt_i32 s16, 14                                       // 00000000095C: BF028E10
	s_cbranch_scc0 16                                          // 000000000960: BF840010 <zluda_unrolled_copy_float+0xa4>
	s_cmp_eq_u32 s16, 15                                       // 000000000964: BF068F10
	s_cbranch_scc0 17                                          // 000000000968: BF840011 <zluda_unrolled_copy_float+0xb0>
	v_lshlrev_b64 v[2:3], 1, v[0:1]                            // 00000000096C: D28F0002 00020081
	v_mov_b32_e32 v4, s3                                       // 000000000974: 7E080203
	v_add_co_u32_e32 v2, vcc, s2, v2                           // 000000000978: 32040402
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc                     // 00000000097C: 38060704
	global_load_ushort v2, v[2:3], off                         // 000000000980: DC488000 027F0002
	s_mov_b64 s[8:9], 0                                        // 000000000988: BE880180
	s_waitcnt vmcnt(0)                                         // 00000000098C: BF8C0F70
	v_lshlrev_b32_e32 v6, 16, v2                               // 000000000990: 240C0490
	s_branch 7                                                 // 000000000994: BF820007 <zluda_unrolled_copy_float+0xb4>
	s_mov_b64 s[10:11], -1                                     // 000000000998: BE8A01C1
	s_mov_b64 s[8:9], 0                                        // 00000000099C: BE880180
	s_branch 21                                                // 0000000009A0: BF820015 <zluda_unrolled_copy_float+0xf8>
	s_mov_b64 s[8:9], 0                                        // 0000000009A4: BE880180
	s_cbranch_execnz 3                                         // 0000000009A8: BF890003 <zluda_unrolled_copy_float+0xb8>
	s_branch 17                                                // 0000000009AC: BF820011 <zluda_unrolled_copy_float+0xf4>
	s_mov_b64 s[8:9], -1                                       // 0000000009B0: BE8801C1
	s_branch 15                                                // 0000000009B4: BF82000F <zluda_unrolled_copy_float+0xf4>
	s_cmp_eq_u32 s16, 7                                        // 0000000009B8: BF068710
	s_cbranch_scc0 12                                          // 0000000009BC: BF84000C <zluda_unrolled_copy_float+0xf0>
	v_lshlrev_b64 v[2:3], 3, v[0:1]                            // 0000000009C0: D28F0002 00020083
	v_mov_b32_e32 v4, s3                                       // 0000000009C8: 7E080203
	v_add_co_u32_e32 v2, vcc, s2, v2                           // 0000000009CC: 32040402
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc                     // 0000000009D0: 38060704
	global_load_dwordx2 v[2:3], v[2:3], off                    // 0000000009D4: DC548000 027F0002
	s_mov_b64 s[8:9], 0                                        // 0000000009DC: BE880180
	s_mov_b64 s[10:11], 0                                      // 0000000009E0: BE8A0180
	s_waitcnt vmcnt(0)                                         // 0000000009E4: BF8C0F70
	v_cvt_f32_f64_e32 v6, v[2:3]                               // 0000000009E8: 7E0C1F02
	s_branch 2                                                 // 0000000009EC: BF820002 <zluda_unrolled_copy_float+0xf8>
	s_mov_b64 s[8:9], -1                                       // 0000000009F0: BE8801C1
	s_mov_b64 s[10:11], 0                                      // 0000000009F4: BE8A0180
	s_and_b64 vcc, exec, s[10:11]                              // 0000000009F8: 86EA0A7E
	s_cbranch_vccnz 64                                         // 0000000009FC: BF870040 <zluda_unrolled_copy_float+0x200>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 000000000A00: D28F0002 00020082
	s_andn2_b64 vcc, exec, s[8:9]                              // 000000000A08: 89EA087E
	s_cbranch_vccz 69                                          // 000000000A0C: BF860045 <zluda_unrolled_copy_float+0x224>
	v_lshlrev_b64 v[4:5], 1, v[0:1]                            // 000000000A10: D28F0004 00020081
	s_andn2_b64 vcc, exec, s[6:7]                              // 000000000A18: 89EA067E
	s_cbranch_vccnz 8                                          // 000000000A1C: BF870008 <zluda_unrolled_copy_float+0x140>
	v_mov_b32_e32 v7, s3                                       // 000000000A20: 7E0E0203
	s_waitcnt vmcnt(0)                                         // 000000000A24: BF8C0F70
	v_add_co_u32_e32 v6, vcc, s2, v4                           // 000000000A28: 320C0802
	v_addc_co_u32_e32 v7, vcc, v7, v5, vcc                     // 000000000A2C: 380E0B07
	global_load_ushort v6, v[6:7], off                         // 000000000A30: DC488000 067F0006
	s_waitcnt vmcnt(0)                                         // 000000000A38: BF8C0F70
	v_cvt_f32_f16_e32 v6, v6                                   // 000000000A3C: 7E0C1706
	s_and_b32 s15, s12, 0xff                                   // 000000000A40: 860FFF0C 000000FF
	s_mov_b64 s[10:11], -1                                     // 000000000A48: BE8A01C1
	s_mov_b64 s[6:7], 0                                        // 000000000A4C: BE860180
	s_cmp_lt_i32 s15, 7                                        // 000000000A50: BF04870F
	s_mov_b64 s[8:9], 0                                        // 000000000A54: BE880180
	s_cbranch_scc0 59                                          // 000000000A58: BF84003B <zluda_unrolled_copy_float+0x248>
	s_and_b64 vcc, exec, s[10:11]                              // 000000000A5C: 86EA0A7E
	s_cbranch_vccnz 88                                         // 000000000A60: BF870058 <zluda_unrolled_copy_float+0x2c4>
	s_andn2_b64 vcc, exec, s[8:9]                              // 000000000A64: 89EA087E
	s_cbranch_vccz 93                                          // 000000000A68: BF86005D <zluda_unrolled_copy_float+0x2e0>
	s_andn2_b64 vcc, exec, s[6:7]                              // 000000000A6C: 89EA067E
	s_cbranch_vccnz 7                                          // 000000000A70: BF870007 <zluda_unrolled_copy_float+0x190>
	s_waitcnt vmcnt(0)                                         // 000000000A74: BF8C0F70
	v_cvt_f16_f32_e32 v3, v6                                   // 000000000A78: 7E061506
	v_mov_b32_e32 v2, s1                                       // 000000000A7C: 7E040201
	v_add_co_u32_e32 v1, vcc, s0, v4                           // 000000000A80: 32020800
	v_addc_co_u32_e32 v2, vcc, v2, v5, vcc                     // 000000000A84: 38040B02
	global_store_short v[1:2], v3, off                         // 000000000A88: DC688000 007F0301
	s_or_b64 exec, exec, s[4:5]                                // 000000000A90: 87FE047E
	v_or_b32_e32 v1, 1, v0                                     // 000000000A94: 28020081
	v_cmp_gt_i32_e32 vcc, s14, v1                              // 000000000A98: 7D88020E
	s_and_saveexec_b64 s[4:5], vcc                             // 000000000A9C: BE84206A
	s_cbranch_execz 152                                        // 000000000AA0: BF880098 <zluda_unrolled_copy_float+0x404>
	s_and_b32 s15, s13, 0xff                                   // 000000000AA4: 860FFF0D 000000FF
	s_cmp_lt_i32 s15, 7                                        // 000000000AAC: BF04870F
	s_mov_b64 s[6:7], 0                                        // 000000000AB0: BE860180
	s_cbranch_scc1 82                                          // 000000000AB4: BF850052 <zluda_unrolled_copy_float+0x300>
	s_and_b32 s16, 0xffff, s15                                 // 000000000AB8: 86100FFF 0000FFFF
	s_cmp_gt_i32 s16, 14                                       // 000000000AC0: BF028E10
	s_cbranch_scc0 81                                          // 000000000AC4: BF840051 <zluda_unrolled_copy_float+0x30c>
	s_cmp_eq_u32 s16, 15                                       // 000000000AC8: BF068F10
	s_cbranch_scc0 82                                          // 000000000ACC: BF840052 <zluda_unrolled_copy_float+0x318>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000AD0: 2202009F
	v_lshlrev_b64 v[1:2], 1, v[0:1]                            // 000000000AD4: D28F0001 00020081
	v_mov_b32_e32 v3, s3                                       // 000000000ADC: 7E060203
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 000000000AE0: 32020202
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000000AE4: 38040503
	global_load_ushort v1, v[1:2], off offset:2                // 000000000AE8: DC488002 017F0001
	s_mov_b64 s[8:9], 0                                        // 000000000AF0: BE880180
	s_waitcnt vmcnt(0)                                         // 000000000AF4: BF8C0F70
	v_lshlrev_b32_e32 v2, 16, v1                               // 000000000AF8: 24040290
	s_branch 71                                                // 000000000AFC: BF820047 <zluda_unrolled_copy_float+0x31c>
	s_and_b32 s8, 0xffff, s15                                  // 000000000B00: 86080FFF 0000FFFF
	s_cmp_lg_u32 s8, 5                                         // 000000000B08: BF078508
	s_mov_b64 s[6:7], -1                                       // 000000000B0C: BE8601C1
	s_cselect_b64 s[8:9], -1, 0                                // 000000000B10: 858880C1
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 000000000B14: D28F0002 00020082
	s_andn2_b64 vcc, exec, s[8:9]                              // 000000000B1C: 89EA087E
	s_cbranch_vccnz 65467                                      // 000000000B20: BF87FFBB <zluda_unrolled_copy_float+0x110>
	v_mov_b32_e32 v5, s3                                       // 000000000B24: 7E0A0203
	v_add_co_u32_e32 v4, vcc, s2, v2                           // 000000000B28: 32080402
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 000000000B2C: 380A0705
	global_load_dword v6, v[4:5], off                          // 000000000B30: DC508000 067F0004
	v_lshlrev_b64 v[4:5], 1, v[0:1]                            // 000000000B38: D28F0004 00020081
	s_cbranch_execz 65463                                      // 000000000B40: BF88FFB7 <zluda_unrolled_copy_float+0x120>
	s_branch 65470                                             // 000000000B44: BF82FFBE <zluda_unrolled_copy_float+0x140>
	s_and_b32 s16, 0xffff, s15                                 // 000000000B48: 86100FFF 0000FFFF
	s_cmp_gt_i32 s16, 14                                       // 000000000B50: BF028E10
	s_cbranch_scc0 11                                          // 000000000B54: BF84000B <zluda_unrolled_copy_float+0x284>
	s_cmp_eq_u32 s16, 15                                       // 000000000B58: BF068F10
	s_mov_b64 s[8:9], -1                                       // 000000000B5C: BE8801C1
	s_cbranch_scc0 7                                           // 000000000B60: BF840007 <zluda_unrolled_copy_float+0x280>
	v_mov_b32_e32 v8, s1                                       // 000000000B64: 7E100201
	v_add_co_u32_e32 v7, vcc, s0, v4                           // 000000000B68: 320E0800
	v_addc_co_u32_e32 v8, vcc, v8, v5, vcc                     // 000000000B6C: 38100B08
	s_waitcnt vmcnt(0)                                         // 000000000B70: BF8C0F70
	global_store_short_d16_hi v[7:8], v6, off                  // 000000000B74: DC6C8000 007F0607
	s_mov_b64 s[8:9], 0                                        // 000000000B7C: BE880180
	s_mov_b64 s[10:11], 0                                      // 000000000B80: BE8A0180
	s_and_b64 vcc, exec, s[10:11]                              // 000000000B84: 86EA0A7E
	s_cbranch_vccz 13                                          // 000000000B88: BF86000D <zluda_unrolled_copy_float+0x2c0>
	s_cmp_eq_u32 s16, 7                                        // 000000000B8C: BF068710
	s_mov_b64 s[8:9], -1                                       // 000000000B90: BE8801C1
	s_cbranch_scc0 10                                          // 000000000B94: BF84000A <zluda_unrolled_copy_float+0x2c0>
	s_waitcnt vmcnt(0)                                         // 000000000B98: BF8C0F70
	v_cvt_f64_f32_e32 v[7:8], v6                               // 000000000B9C: 7E0E2106
	v_lshlrev_b64 v[9:10], 3, v[0:1]                           // 000000000BA0: D28F0009 00020083
	v_mov_b32_e32 v1, s1                                       // 000000000BA8: 7E020201
	v_add_co_u32_e32 v9, vcc, s0, v9                           // 000000000BAC: 32121200
	v_addc_co_u32_e32 v10, vcc, v1, v10, vcc                   // 000000000BB0: 38141501
	global_store_dwordx2 v[9:10], v[7:8], off                  // 000000000BB4: DC748000 007F0709
	s_mov_b64 s[8:9], 0                                        // 000000000BBC: BE880180
	s_branch 65448                                             // 000000000BC0: BF82FFA8 <zluda_unrolled_copy_float+0x164>
	s_and_b32 s8, 0xffff, s15                                  // 000000000BC4: 86080FFF 0000FFFF
	s_cmp_lg_u32 s8, 5                                         // 000000000BCC: BF078508
	s_mov_b64 s[6:7], -1                                       // 000000000BD0: BE8601C1
	s_cselect_b64 s[8:9], -1, 0                                // 000000000BD4: 858880C1
	s_andn2_b64 vcc, exec, s[8:9]                              // 000000000BD8: 89EA087E
	s_cbranch_vccnz 65443                                      // 000000000BDC: BF87FFA3 <zluda_unrolled_copy_float+0x16c>
	v_mov_b32_e32 v7, s1                                       // 000000000BE0: 7E0E0201
	v_add_co_u32_e32 v1, vcc, s0, v2                           // 000000000BE4: 32020400
	v_addc_co_u32_e32 v2, vcc, v7, v3, vcc                     // 000000000BE8: 38040707
	s_waitcnt vmcnt(0)                                         // 000000000BEC: BF8C0F70
	global_store_dword v[1:2], v6, off                         // 000000000BF0: DC708000 007F0601
	s_cbranch_execz 65438                                      // 000000000BF8: BF88FF9E <zluda_unrolled_copy_float+0x174>
	s_branch 65444                                             // 000000000BFC: BF82FFA4 <zluda_unrolled_copy_float+0x190>
	s_mov_b64 s[8:9], 0                                        // 000000000C00: BE880180
	s_cbranch_execnz 20                                        // 000000000C04: BF890014 <zluda_unrolled_copy_float+0x358>
	s_branch 24                                                // 000000000C08: BF820018 <zluda_unrolled_copy_float+0x36c>
	s_mov_b64 s[8:9], 0                                        // 000000000C0C: BE880180
	s_cbranch_execnz 3                                         // 000000000C10: BF890003 <zluda_unrolled_copy_float+0x320>
	s_branch 15                                                // 000000000C14: BF82000F <zluda_unrolled_copy_float+0x354>
	s_mov_b64 s[8:9], -1                                       // 000000000C18: BE8801C1
	s_branch 13                                                // 000000000C1C: BF82000D <zluda_unrolled_copy_float+0x354>
	s_cmp_eq_u32 s16, 7                                        // 000000000C20: BF068710
	s_cbranch_scc0 153                                         // 000000000C24: BF840099 <zluda_unrolled_copy_float+0x58c>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000C28: 2202009F
	v_lshlrev_b64 v[1:2], 3, v[0:1]                            // 000000000C2C: D28F0001 00020083
	v_mov_b32_e32 v3, s3                                       // 000000000C34: 7E060203
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 000000000C38: 32020202
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000000C3C: 38040503
	global_load_dwordx2 v[1:2], v[1:2], off offset:8           // 000000000C40: DC548008 017F0001
	s_mov_b64 s[8:9], 0                                        // 000000000C48: BE880180
	s_waitcnt vmcnt(0)                                         // 000000000C4C: BF8C0F70
	v_cvt_f32_f64_e32 v2, v[1:2]                               // 000000000C50: 7E041F01
	s_branch 5                                                 // 000000000C54: BF820005 <zluda_unrolled_copy_float+0x36c>
	s_and_b32 s8, 0xffff, s15                                  // 000000000C58: 86080FFF 0000FFFF
	s_cmp_lg_u32 s8, 5                                         // 000000000C60: BF078508
	s_mov_b64 s[6:7], -1                                       // 000000000C64: BE8601C1
	s_cselect_b64 s[8:9], -1, 0                                // 000000000C68: 858880C1
	s_andn2_b64 vcc, exec, s[8:9]                              // 000000000C6C: 89EA087E
	s_cbranch_vccz 65                                          // 000000000C70: BF860041 <zluda_unrolled_copy_float+0x478>
	s_andn2_b64 vcc, exec, s[6:7]                              // 000000000C74: 89EA067E
	s_cbranch_vccnz 11                                         // 000000000C78: BF87000B <zluda_unrolled_copy_float+0x3a8>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000C7C: 2202009F
	s_waitcnt vmcnt(0)                                         // 000000000C80: BF8C0F70
	v_lshlrev_b64 v[1:2], 1, v[0:1]                            // 000000000C84: D28F0001 00020081
	v_mov_b32_e32 v3, s3                                       // 000000000C8C: 7E060203
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 000000000C90: 32020202
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000000C94: 38040503
	global_load_ushort v1, v[1:2], off offset:2                // 000000000C98: DC488002 017F0001
	s_waitcnt vmcnt(0)                                         // 000000000CA0: BF8C0F70
	v_cvt_f32_f16_e32 v2, v1                                   // 000000000CA4: 7E041701
	s_and_b32 s15, s12, 0xff                                   // 000000000CA8: 860FFF0C 000000FF
	s_mov_b64 s[10:11], -1                                     // 000000000CB0: BE8A01C1
	s_mov_b64 s[6:7], 0                                        // 000000000CB4: BE860180
	s_cmp_lt_i32 s15, 7                                        // 000000000CB8: BF04870F
	s_mov_b64 s[8:9], 0                                        // 000000000CBC: BE880180
	s_cbranch_scc0 55                                          // 000000000CC0: BF840037 <zluda_unrolled_copy_float+0x4a0>
	s_and_b64 vcc, exec, s[10:11]                              // 000000000CC4: 86EA0A7E
	s_cbranch_vccnz 88                                         // 000000000CC8: BF870058 <zluda_unrolled_copy_float+0x52c>
	s_andn2_b64 vcc, exec, s[8:9]                              // 000000000CCC: 89EA087E
	s_cbranch_vccz 93                                          // 000000000CD0: BF86005D <zluda_unrolled_copy_float+0x548>
	s_andn2_b64 vcc, exec, s[6:7]                              // 000000000CD4: 89EA067E
	s_cbranch_vccnz 10                                         // 000000000CD8: BF87000A <zluda_unrolled_copy_float+0x404>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000CDC: 2202009F
	s_waitcnt vmcnt(0)                                         // 000000000CE0: BF8C0F70
	v_cvt_f16_f32_e32 v3, v2                                   // 000000000CE4: 7E061502
	v_lshlrev_b64 v[1:2], 1, v[0:1]                            // 000000000CE8: D28F0001 00020081
	v_mov_b32_e32 v4, s1                                       // 000000000CF0: 7E080201
	v_add_co_u32_e32 v1, vcc, s0, v1                           // 000000000CF4: 32020200
	v_addc_co_u32_e32 v2, vcc, v4, v2, vcc                     // 000000000CF8: 38040504
	global_store_short v[1:2], v3, off offset:2                // 000000000CFC: DC688002 007F0301
	s_or_b64 exec, exec, s[4:5]                                // 000000000D04: 87FE047E
	v_or_b32_e32 v1, 2, v0                                     // 000000000D08: 28020082
	v_cmp_gt_i32_e32 vcc, s14, v1                              // 000000000D0C: 7D88020E
	s_and_saveexec_b64 s[4:5], vcc                             // 000000000D10: BE84206A
	s_cbranch_execz 155                                        // 000000000D14: BF88009B <zluda_unrolled_copy_float+0x684>
	s_and_b32 s15, s13, 0xff                                   // 000000000D18: 860FFF0D 000000FF
	s_cmp_lt_i32 s15, 7                                        // 000000000D20: BF04870F
	s_mov_b64 s[6:7], 0                                        // 000000000D24: BE860180
	s_cbranch_scc1 82                                          // 000000000D28: BF850052 <zluda_unrolled_copy_float+0x574>
	s_and_b32 s16, 0xffff, s15                                 // 000000000D2C: 86100FFF 0000FFFF
	s_cmp_gt_i32 s16, 14                                       // 000000000D34: BF028E10
	s_cbranch_scc0 81                                          // 000000000D38: BF840051 <zluda_unrolled_copy_float+0x580>
	s_cmp_eq_u32 s16, 15                                       // 000000000D3C: BF068F10
	s_cbranch_scc0 84                                          // 000000000D40: BF840054 <zluda_unrolled_copy_float+0x594>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000D44: 2202009F
	s_waitcnt vmcnt(0)                                         // 000000000D48: BF8C0F70
	v_lshlrev_b64 v[1:2], 1, v[0:1]                            // 000000000D4C: D28F0001 00020081
	v_mov_b32_e32 v3, s3                                       // 000000000D54: 7E060203
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 000000000D58: 32020202
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000000D5C: 38040503
	global_load_ushort v1, v[1:2], off offset:4                // 000000000D60: DC488004 017F0001
	s_mov_b64 s[8:9], 0                                        // 000000000D68: BE880180
	s_waitcnt vmcnt(0)                                         // 000000000D6C: BF8C0F70
	v_lshlrev_b32_e32 v2, 16, v1                               // 000000000D70: 24040290
	s_branch 72                                                // 000000000D74: BF820048 <zluda_unrolled_copy_float+0x598>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000D78: 2202009F
	v_lshlrev_b64 v[1:2], 2, v[0:1]                            // 000000000D7C: D28F0001 00020082
	v_mov_b32_e32 v3, s3                                       // 000000000D84: 7E060203
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 000000000D88: 32020202
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000000D8C: 38040503
	global_load_dword v2, v[1:2], off offset:4                 // 000000000D90: DC508004 027F0001
	s_cbranch_execz 65464                                      // 000000000D98: BF88FFB8 <zluda_unrolled_copy_float+0x37c>
	s_branch 65474                                             // 000000000D9C: BF82FFC2 <zluda_unrolled_copy_float+0x3a8>
	s_and_b32 s16, 0xffff, s15                                 // 000000000DA0: 86100FFF 0000FFFF
	s_cmp_gt_i32 s16, 14                                       // 000000000DA8: BF028E10
	s_cbranch_scc0 14                                          // 000000000DAC: BF84000E <zluda_unrolled_copy_float+0x4e8>
	s_cmp_eq_u32 s16, 15                                       // 000000000DB0: BF068F10
	s_mov_b64 s[8:9], -1                                       // 000000000DB4: BE8801C1
	s_cbranch_scc0 10                                          // 000000000DB8: BF84000A <zluda_unrolled_copy_float+0x4e4>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000DBC: 2202009F
	v_lshlrev_b64 v[3:4], 1, v[0:1]                            // 000000000DC0: D28F0003 00020081
	v_mov_b32_e32 v1, s1                                       // 000000000DC8: 7E020201
	v_add_co_u32_e32 v3, vcc, s0, v3                           // 000000000DCC: 32060600
	v_addc_co_u32_e32 v4, vcc, v1, v4, vcc                     // 000000000DD0: 38080901
	s_waitcnt vmcnt(0)                                         // 000000000DD4: BF8C0F70
	global_store_short_d16_hi v[3:4], v2, off offset:2         // 000000000DD8: DC6C8002 007F0203
	s_mov_b64 s[8:9], 0                                        // 000000000DE0: BE880180
	s_mov_b64 s[10:11], 0                                      // 000000000DE4: BE8A0180
	s_and_b64 vcc, exec, s[10:11]                              // 000000000DE8: 86EA0A7E
	s_cbranch_vccz 14                                          // 000000000DEC: BF86000E <zluda_unrolled_copy_float+0x528>
	s_cmp_eq_u32 s16, 7                                        // 000000000DF0: BF068710
	s_mov_b64 s[8:9], -1                                       // 000000000DF4: BE8801C1
	s_cbranch_scc0 11                                          // 000000000DF8: BF84000B <zluda_unrolled_copy_float+0x528>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000DFC: 2202009F
	s_waitcnt vmcnt(0)                                         // 000000000E00: BF8C0F70
	v_cvt_f64_f32_e32 v[3:4], v2                               // 000000000E04: 7E062102
	v_lshlrev_b64 v[5:6], 3, v[0:1]                            // 000000000E08: D28F0005 00020083
	v_mov_b32_e32 v1, s1                                       // 000000000E10: 7E020201
	v_add_co_u32_e32 v5, vcc, s0, v5                           // 000000000E14: 320A0A00
	v_addc_co_u32_e32 v6, vcc, v1, v6, vcc                     // 000000000E18: 380C0D01
	global_store_dwordx2 v[5:6], v[3:4], off offset:8          // 000000000E1C: DC748008 007F0305
	s_mov_b64 s[8:9], 0                                        // 000000000E24: BE880180
	s_branch 65448                                             // 000000000E28: BF82FFA8 <zluda_unrolled_copy_float+0x3cc>
	s_and_b32 s8, 0xffff, s15                                  // 000000000E2C: 86080FFF 0000FFFF
	s_cmp_lg_u32 s8, 5                                         // 000000000E34: BF078508
	s_mov_b64 s[6:7], -1                                       // 000000000E38: BE8601C1
	s_cselect_b64 s[8:9], -1, 0                                // 000000000E3C: 858880C1
	s_andn2_b64 vcc, exec, s[8:9]                              // 000000000E40: 89EA087E
	s_cbranch_vccnz 65443                                      // 000000000E44: BF87FFA3 <zluda_unrolled_copy_float+0x3d4>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000E48: 2202009F
	v_lshlrev_b64 v[3:4], 2, v[0:1]                            // 000000000E4C: D28F0003 00020082
	v_mov_b32_e32 v1, s1                                       // 000000000E54: 7E020201
	v_add_co_u32_e32 v3, vcc, s0, v3                           // 000000000E58: 32060600
	v_addc_co_u32_e32 v4, vcc, v1, v4, vcc                     // 000000000E5C: 38080901
	s_waitcnt vmcnt(0)                                         // 000000000E60: BF8C0F70
	global_store_dword v[3:4], v2, off offset:4                // 000000000E64: DC708004 007F0203
	s_cbranch_execz 65435                                      // 000000000E6C: BF88FF9B <zluda_unrolled_copy_float+0x3dc>
	s_branch 65444                                             // 000000000E70: BF82FFA4 <zluda_unrolled_copy_float+0x404>
	s_mov_b64 s[8:9], 0                                        // 000000000E74: BE880180
	s_cbranch_execnz 23                                        // 000000000E78: BF890017 <zluda_unrolled_copy_float+0x5d8>
	s_branch 27                                                // 000000000E7C: BF82001B <zluda_unrolled_copy_float+0x5ec>
	s_mov_b64 s[8:9], 0                                        // 000000000E80: BE880180
	s_cbranch_execnz 5                                         // 000000000E84: BF890005 <zluda_unrolled_copy_float+0x59c>
	s_branch 18                                                // 000000000E88: BF820012 <zluda_unrolled_copy_float+0x5d4>
	s_mov_b64 s[8:9], -1                                       // 000000000E8C: BE8801C1
	s_branch 65398                                             // 000000000E90: BF82FF76 <zluda_unrolled_copy_float+0x36c>
	s_mov_b64 s[8:9], -1                                       // 000000000E94: BE8801C1
	s_branch 14                                                // 000000000E98: BF82000E <zluda_unrolled_copy_float+0x5d4>
	s_cmp_eq_u32 s16, 7                                        // 000000000E9C: BF068710
	s_cbranch_scc0 155                                         // 000000000EA0: BF84009B <zluda_unrolled_copy_float+0x810>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000EA4: 2202009F
	s_waitcnt vmcnt(0)                                         // 000000000EA8: BF8C0F70
	v_lshlrev_b64 v[1:2], 3, v[0:1]                            // 000000000EAC: D28F0001 00020083
	v_mov_b32_e32 v3, s3                                       // 000000000EB4: 7E060203
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 000000000EB8: 32020202
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000000EBC: 38040503
	global_load_dwordx2 v[1:2], v[1:2], off offset:16          // 000000000EC0: DC548010 017F0001
	s_mov_b64 s[8:9], 0                                        // 000000000EC8: BE880180
	s_waitcnt vmcnt(0)                                         // 000000000ECC: BF8C0F70
	v_cvt_f32_f64_e32 v2, v[1:2]                               // 000000000ED0: 7E041F01
	s_branch 5                                                 // 000000000ED4: BF820005 <zluda_unrolled_copy_float+0x5ec>
	s_and_b32 s8, 0xffff, s15                                  // 000000000ED8: 86080FFF 0000FFFF
	s_cmp_lg_u32 s8, 5                                         // 000000000EE0: BF078508
	s_mov_b64 s[6:7], -1                                       // 000000000EE4: BE8601C1
	s_cselect_b64 s[8:9], -1, 0                                // 000000000EE8: 858880C1
	s_andn2_b64 vcc, exec, s[8:9]                              // 000000000EEC: 89EA087E
	s_cbranch_vccz 65                                          // 000000000EF0: BF860041 <zluda_unrolled_copy_float+0x6f8>
	s_andn2_b64 vcc, exec, s[6:7]                              // 000000000EF4: 89EA067E
	s_cbranch_vccnz 11                                         // 000000000EF8: BF87000B <zluda_unrolled_copy_float+0x628>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000EFC: 2202009F
	s_waitcnt vmcnt(0)                                         // 000000000F00: BF8C0F70
	v_lshlrev_b64 v[1:2], 1, v[0:1]                            // 000000000F04: D28F0001 00020081
	v_mov_b32_e32 v3, s3                                       // 000000000F0C: 7E060203
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 000000000F10: 32020202
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000000F14: 38040503
	global_load_ushort v1, v[1:2], off offset:4                // 000000000F18: DC488004 017F0001
	s_waitcnt vmcnt(0)                                         // 000000000F20: BF8C0F70
	v_cvt_f32_f16_e32 v2, v1                                   // 000000000F24: 7E041701
	s_and_b32 s15, s12, 0xff                                   // 000000000F28: 860FFF0C 000000FF
	s_mov_b64 s[10:11], -1                                     // 000000000F30: BE8A01C1
	s_mov_b64 s[6:7], 0                                        // 000000000F34: BE860180
	s_cmp_lt_i32 s15, 7                                        // 000000000F38: BF04870F
	s_mov_b64 s[8:9], 0                                        // 000000000F3C: BE880180
	s_cbranch_scc0 56                                          // 000000000F40: BF840038 <zluda_unrolled_copy_float+0x724>
	s_and_b64 vcc, exec, s[10:11]                              // 000000000F44: 86EA0A7E
	s_cbranch_vccnz 89                                         // 000000000F48: BF870059 <zluda_unrolled_copy_float+0x7b0>
	s_andn2_b64 vcc, exec, s[8:9]                              // 000000000F4C: 89EA087E
	s_cbranch_vccz 94                                          // 000000000F50: BF86005E <zluda_unrolled_copy_float+0x7cc>
	s_andn2_b64 vcc, exec, s[6:7]                              // 000000000F54: 89EA067E
	s_cbranch_vccnz 10                                         // 000000000F58: BF87000A <zluda_unrolled_copy_float+0x684>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000F5C: 2202009F
	s_waitcnt vmcnt(0)                                         // 000000000F60: BF8C0F70
	v_cvt_f16_f32_e32 v3, v2                                   // 000000000F64: 7E061502
	v_lshlrev_b64 v[1:2], 1, v[0:1]                            // 000000000F68: D28F0001 00020081
	v_mov_b32_e32 v4, s1                                       // 000000000F70: 7E080201
	v_add_co_u32_e32 v1, vcc, s0, v1                           // 000000000F74: 32020200
	v_addc_co_u32_e32 v2, vcc, v4, v2, vcc                     // 000000000F78: 38040504
	global_store_short v[1:2], v3, off offset:4                // 000000000F7C: DC688004 007F0301
	s_or_b64 exec, exec, s[4:5]                                // 000000000F84: 87FE047E
	v_or_b32_e32 v1, 3, v0                                     // 000000000F88: 28020083
	v_cmp_gt_i32_e32 vcc, s14, v1                              // 000000000F8C: 7D88020E
	s_and_saveexec_b64 s[4:5], vcc                             // 000000000F90: BE84206A
	s_cbranch_execz 155                                        // 000000000F94: BF88009B <zluda_unrolled_copy_float+0x904>
	s_and_b32 s10, s13, 0xff                                   // 000000000F98: 860AFF0D 000000FF
	s_cmp_lt_i32 s10, 7                                        // 000000000FA0: BF04870A
	s_mov_b64 s[4:5], 0                                        // 000000000FA4: BE840180
	s_cbranch_scc1 83                                          // 000000000FA8: BF850053 <zluda_unrolled_copy_float+0x7f8>
	s_and_b32 s11, 0xffff, s10                                 // 000000000FAC: 860B0AFF 0000FFFF
	s_cmp_gt_i32 s11, 14                                       // 000000000FB4: BF028E0B
	s_cbranch_scc0 82                                          // 000000000FB8: BF840052 <zluda_unrolled_copy_float+0x804>
	s_cmp_eq_u32 s11, 15                                       // 000000000FBC: BF068F0B
	s_cbranch_scc0 85                                          // 000000000FC0: BF840055 <zluda_unrolled_copy_float+0x818>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000FC4: 2202009F
	s_waitcnt vmcnt(0)                                         // 000000000FC8: BF8C0F70
	v_lshlrev_b64 v[1:2], 1, v[0:1]                            // 000000000FCC: D28F0001 00020081
	v_mov_b32_e32 v3, s3                                       // 000000000FD4: 7E060203
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 000000000FD8: 32020202
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000000FDC: 38040503
	global_load_ushort v1, v[1:2], off offset:6                // 000000000FE0: DC488006 017F0001
	s_mov_b64 s[6:7], 0                                        // 000000000FE8: BE860180
	s_waitcnt vmcnt(0)                                         // 000000000FEC: BF8C0F70
	v_lshlrev_b32_e32 v2, 16, v1                               // 000000000FF0: 24040290
	s_branch 73                                                // 000000000FF4: BF820049 <zluda_unrolled_copy_float+0x81c>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000000FF8: 2202009F
	s_waitcnt vmcnt(0)                                         // 000000000FFC: BF8C0F70
	v_lshlrev_b64 v[1:2], 2, v[0:1]                            // 000000001000: D28F0001 00020082
	v_mov_b32_e32 v3, s3                                       // 000000001008: 7E060203
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 00000000100C: 32020202
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000001010: 38040503
	global_load_dword v2, v[1:2], off offset:8                 // 000000001014: DC508008 027F0001
	s_cbranch_execz 65463                                      // 00000000101C: BF88FFB7 <zluda_unrolled_copy_float+0x5fc>
	s_branch 65473                                             // 000000001020: BF82FFC1 <zluda_unrolled_copy_float+0x628>
	s_and_b32 s16, 0xffff, s15                                 // 000000001024: 86100FFF 0000FFFF
	s_cmp_gt_i32 s16, 14                                       // 00000000102C: BF028E10
	s_cbranch_scc0 14                                          // 000000001030: BF84000E <zluda_unrolled_copy_float+0x76c>
	s_cmp_eq_u32 s16, 15                                       // 000000001034: BF068F10
	s_mov_b64 s[8:9], -1                                       // 000000001038: BE8801C1
	s_cbranch_scc0 10                                          // 00000000103C: BF84000A <zluda_unrolled_copy_float+0x768>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000001040: 2202009F
	v_lshlrev_b64 v[3:4], 1, v[0:1]                            // 000000001044: D28F0003 00020081
	v_mov_b32_e32 v1, s1                                       // 00000000104C: 7E020201
	v_add_co_u32_e32 v3, vcc, s0, v3                           // 000000001050: 32060600
	v_addc_co_u32_e32 v4, vcc, v1, v4, vcc                     // 000000001054: 38080901
	s_waitcnt vmcnt(0)                                         // 000000001058: BF8C0F70
	global_store_short_d16_hi v[3:4], v2, off offset:4         // 00000000105C: DC6C8004 007F0203
	s_mov_b64 s[8:9], 0                                        // 000000001064: BE880180
	s_mov_b64 s[10:11], 0                                      // 000000001068: BE8A0180
	s_and_b64 vcc, exec, s[10:11]                              // 00000000106C: 86EA0A7E
	s_cbranch_vccz 14                                          // 000000001070: BF86000E <zluda_unrolled_copy_float+0x7ac>
	s_cmp_eq_u32 s16, 7                                        // 000000001074: BF068710
	s_mov_b64 s[8:9], -1                                       // 000000001078: BE8801C1
	s_cbranch_scc0 11                                          // 00000000107C: BF84000B <zluda_unrolled_copy_float+0x7ac>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000001080: 2202009F
	s_waitcnt vmcnt(0)                                         // 000000001084: BF8C0F70
	v_cvt_f64_f32_e32 v[3:4], v2                               // 000000001088: 7E062102
	v_lshlrev_b64 v[5:6], 3, v[0:1]                            // 00000000108C: D28F0005 00020083
	v_mov_b32_e32 v1, s1                                       // 000000001094: 7E020201
	v_add_co_u32_e32 v5, vcc, s0, v5                           // 000000001098: 320A0A00
	v_addc_co_u32_e32 v6, vcc, v1, v6, vcc                     // 00000000109C: 380C0D01
	global_store_dwordx2 v[5:6], v[3:4], off offset:16         // 0000000010A0: DC748010 007F0305
	s_mov_b64 s[8:9], 0                                        // 0000000010A8: BE880180
	s_branch 65447                                             // 0000000010AC: BF82FFA7 <zluda_unrolled_copy_float+0x64c>
	s_and_b32 s8, 0xffff, s15                                  // 0000000010B0: 86080FFF 0000FFFF
	s_cmp_lg_u32 s8, 5                                         // 0000000010B8: BF078508
	s_mov_b64 s[6:7], -1                                       // 0000000010BC: BE8601C1
	s_cselect_b64 s[8:9], -1, 0                                // 0000000010C0: 858880C1
	s_andn2_b64 vcc, exec, s[8:9]                              // 0000000010C4: 89EA087E
	s_cbranch_vccnz 65442                                      // 0000000010C8: BF87FFA2 <zluda_unrolled_copy_float+0x654>
	v_ashrrev_i32_e32 v1, 31, v0                               // 0000000010CC: 2202009F
	v_lshlrev_b64 v[3:4], 2, v[0:1]                            // 0000000010D0: D28F0003 00020082
	v_mov_b32_e32 v1, s1                                       // 0000000010D8: 7E020201
	v_add_co_u32_e32 v3, vcc, s0, v3                           // 0000000010DC: 32060600
	v_addc_co_u32_e32 v4, vcc, v1, v4, vcc                     // 0000000010E0: 38080901
	s_waitcnt vmcnt(0)                                         // 0000000010E4: BF8C0F70
	global_store_dword v[3:4], v2, off offset:8                // 0000000010E8: DC708008 007F0203
	s_cbranch_execz 65434                                      // 0000000010F0: BF88FF9A <zluda_unrolled_copy_float+0x65c>
	s_branch 65443                                             // 0000000010F4: BF82FFA3 <zluda_unrolled_copy_float+0x684>
	s_mov_b64 s[6:7], 0                                        // 0000000010F8: BE860180
	s_cbranch_execnz 23                                        // 0000000010FC: BF890017 <zluda_unrolled_copy_float+0x85c>
	s_branch 27                                                // 000000001100: BF82001B <zluda_unrolled_copy_float+0x870>
	s_mov_b64 s[6:7], 0                                        // 000000001104: BE860180
	s_cbranch_execnz 5                                         // 000000001108: BF890005 <zluda_unrolled_copy_float+0x820>
	s_branch 18                                                // 00000000110C: BF820012 <zluda_unrolled_copy_float+0x858>
	s_mov_b64 s[8:9], -1                                       // 000000001110: BE8801C1
	s_branch 65397                                             // 000000001114: BF82FF75 <zluda_unrolled_copy_float+0x5ec>
	s_mov_b64 s[6:7], -1                                       // 000000001118: BE8601C1
	s_branch 14                                                // 00000000111C: BF82000E <zluda_unrolled_copy_float+0x858>
	s_cmp_eq_u32 s11, 7                                        // 000000001120: BF06870B
	s_cbranch_scc0 116                                         // 000000001124: BF840074 <zluda_unrolled_copy_float+0x9f8>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000001128: 2202009F
	s_waitcnt vmcnt(0)                                         // 00000000112C: BF8C0F70
	v_lshlrev_b64 v[1:2], 3, v[0:1]                            // 000000001130: D28F0001 00020083
	v_mov_b32_e32 v3, s3                                       // 000000001138: 7E060203
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 00000000113C: 32020202
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000001140: 38040503
	global_load_dwordx2 v[1:2], v[1:2], off offset:24          // 000000001144: DC548018 017F0001
	s_mov_b64 s[6:7], 0                                        // 00000000114C: BE860180
	s_waitcnt vmcnt(0)                                         // 000000001150: BF8C0F70
	v_cvt_f32_f64_e32 v2, v[1:2]                               // 000000001154: 7E041F01
	s_branch 5                                                 // 000000001158: BF820005 <zluda_unrolled_copy_float+0x870>
	s_and_b32 s6, 0xffff, s10                                  // 00000000115C: 86060AFF 0000FFFF
	s_cmp_lg_u32 s6, 5                                         // 000000001164: BF078506
	s_mov_b64 s[4:5], -1                                       // 000000001168: BE8401C1
	s_cselect_b64 s[6:7], -1, 0                                // 00000000116C: 858680C1
	s_andn2_b64 vcc, exec, s[6:7]                              // 000000001170: 89EA067E
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000001174: 2202009F
	s_cbranch_vccz 35                                          // 000000001178: BF860023 <zluda_unrolled_copy_float+0x908>
	s_andn2_b64 vcc, exec, s[4:5]                              // 00000000117C: 89EA047E
	s_cbranch_vccnz 10                                         // 000000001180: BF87000A <zluda_unrolled_copy_float+0x8ac>
	s_waitcnt vmcnt(0)                                         // 000000001184: BF8C0F70
	v_lshlrev_b64 v[2:3], 1, v[0:1]                            // 000000001188: D28F0002 00020081
	v_mov_b32_e32 v4, s3                                       // 000000001190: 7E080203
	v_add_co_u32_e32 v2, vcc, s2, v2                           // 000000001194: 32040402
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc                     // 000000001198: 38060704
	global_load_ushort v2, v[2:3], off offset:6                // 00000000119C: DC488006 027F0002
	s_waitcnt vmcnt(0)                                         // 0000000011A4: BF8C0F70
	v_cvt_f32_f16_e32 v2, v2                                   // 0000000011A8: 7E041702
	s_and_b32 s8, s12, 0xff                                    // 0000000011AC: 8608FF0C 000000FF
	s_mov_b64 s[6:7], -1                                       // 0000000011B4: BE8601C1
	s_mov_b64 s[2:3], 0                                        // 0000000011B8: BE820180
	s_cmp_lt_i32 s8, 7                                         // 0000000011BC: BF048708
	s_mov_b64 s[4:5], 0                                        // 0000000011C0: BE840180
	s_cbranch_scc0 26                                          // 0000000011C4: BF84001A <zluda_unrolled_copy_float+0x930>
	s_and_b64 vcc, exec, s[6:7]                                // 0000000011C8: 86EA067E
	s_cbranch_vccnz 57                                         // 0000000011CC: BF870039 <zluda_unrolled_copy_float+0x9b4>
	s_andn2_b64 vcc, exec, s[4:5]                              // 0000000011D0: 89EA047E
	s_cbranch_vccz 62                                          // 0000000011D4: BF86003E <zluda_unrolled_copy_float+0x9d0>
	s_andn2_b64 vcc, exec, s[2:3]                              // 0000000011D8: 89EA027E
	s_cbranch_vccnz 9                                          // 0000000011DC: BF870009 <zluda_unrolled_copy_float+0x904>
	s_waitcnt vmcnt(0)                                         // 0000000011E0: BF8C0F70
	v_cvt_f16_f32_e32 v2, v2                                   // 0000000011E4: 7E041502
	v_lshlrev_b64 v[0:1], 1, v[0:1]                            // 0000000011E8: D28F0000 00020081
	v_mov_b32_e32 v3, s1                                       // 0000000011F0: 7E060201
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 0000000011F4: 32000000
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 0000000011F8: 38020303
	global_store_short v[0:1], v2, off offset:6                // 0000000011FC: DC688006 007F0200
	s_endpgm                                                   // 000000001204: BF810000
	s_waitcnt vmcnt(0)                                         // 000000001208: BF8C0F70
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 00000000120C: D28F0002 00020082
	v_mov_b32_e32 v4, s3                                       // 000000001214: 7E080203
	v_add_co_u32_e32 v2, vcc, s2, v2                           // 000000001218: 32040402
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc                     // 00000000121C: 38060704
	global_load_dword v2, v[2:3], off offset:12                // 000000001220: DC50800C 027F0002
	s_cbranch_execz 65494                                      // 000000001228: BF88FFD6 <zluda_unrolled_copy_float+0x884>
	s_branch 65503                                             // 00000000122C: BF82FFDF <zluda_unrolled_copy_float+0x8ac>
	s_and_b32 s9, 0xffff, s8                                   // 000000001230: 860908FF 0000FFFF
	s_cmp_gt_i32 s9, 14                                        // 000000001238: BF028E09
	s_cbranch_scc0 13                                          // 00000000123C: BF84000D <zluda_unrolled_copy_float+0x974>
	s_cmp_eq_u32 s9, 15                                        // 000000001240: BF068F09
	s_mov_b64 s[4:5], -1                                       // 000000001244: BE8401C1
	s_cbranch_scc0 9                                           // 000000001248: BF840009 <zluda_unrolled_copy_float+0x970>
	v_lshlrev_b64 v[3:4], 1, v[0:1]                            // 00000000124C: D28F0003 00020081
	v_mov_b32_e32 v5, s1                                       // 000000001254: 7E0A0201
	v_add_co_u32_e32 v3, vcc, s0, v3                           // 000000001258: 32060600
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc                     // 00000000125C: 38080905
	s_waitcnt vmcnt(0)                                         // 000000001260: BF8C0F70
	global_store_short_d16_hi v[3:4], v2, off offset:6         // 000000001264: DC6C8006 007F0203
	s_mov_b64 s[4:5], 0                                        // 00000000126C: BE840180
	s_mov_b64 s[6:7], 0                                        // 000000001270: BE860180
	s_and_b64 vcc, exec, s[6:7]                                // 000000001274: 86EA067E
	s_cbranch_vccz 13                                          // 000000001278: BF86000D <zluda_unrolled_copy_float+0x9b0>
	s_cmp_eq_u32 s9, 7                                         // 00000000127C: BF068709
	s_mov_b64 s[4:5], -1                                       // 000000001280: BE8401C1
	s_cbranch_scc0 10                                          // 000000001284: BF84000A <zluda_unrolled_copy_float+0x9b0>
	s_waitcnt vmcnt(0)                                         // 000000001288: BF8C0F70
	v_cvt_f64_f32_e32 v[3:4], v2                               // 00000000128C: 7E062102
	v_lshlrev_b64 v[5:6], 3, v[0:1]                            // 000000001290: D28F0005 00020083
	v_mov_b32_e32 v7, s1                                       // 000000001298: 7E0E0201
	v_add_co_u32_e32 v5, vcc, s0, v5                           // 00000000129C: 320A0A00
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc                     // 0000000012A0: 380C0D07
	global_store_dwordx2 v[5:6], v[3:4], off offset:24         // 0000000012A4: DC748018 007F0305
	s_mov_b64 s[4:5], 0                                        // 0000000012AC: BE840180
	s_branch 65479                                             // 0000000012B0: BF82FFC7 <zluda_unrolled_copy_float+0x8d0>
	s_and_b32 s4, 0xffff, s8                                   // 0000000012B4: 860408FF 0000FFFF
	s_cmp_lg_u32 s4, 5                                         // 0000000012BC: BF078504
	s_mov_b64 s[2:3], -1                                       // 0000000012C0: BE8201C1
	s_cselect_b64 s[4:5], -1, 0                                // 0000000012C4: 858480C1
	s_andn2_b64 vcc, exec, s[4:5]                              // 0000000012C8: 89EA047E
	s_cbranch_vccnz 65474                                      // 0000000012CC: BF87FFC2 <zluda_unrolled_copy_float+0x8d8>
	v_lshlrev_b64 v[3:4], 2, v[0:1]                            // 0000000012D0: D28F0003 00020082
	v_mov_b32_e32 v5, s1                                       // 0000000012D8: 7E0A0201
	v_add_co_u32_e32 v3, vcc, s0, v3                           // 0000000012DC: 32060600
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc                     // 0000000012E0: 38080905
	s_waitcnt vmcnt(0)                                         // 0000000012E4: BF8C0F70
	global_store_dword v[3:4], v2, off offset:12               // 0000000012E8: DC70800C 007F0203
	s_cbranch_execz 65467                                      // 0000000012F0: BF88FFBB <zluda_unrolled_copy_float+0x8e0>
	s_branch 65475                                             // 0000000012F4: BF82FFC3 <zluda_unrolled_copy_float+0x904>
	s_mov_b64 s[6:7], -1                                       // 0000000012F8: BE8601C1
	s_branch 65436                                             // 0000000012FC: BF82FF9C <zluda_unrolled_copy_float+0x870>

0000000000001300 <zluda_vec4_sq_float>:
	s_load_dword s7, s[4:5], 0x0                               // 000000001300: C00201C2 00000000
	s_load_dwordx4 s[0:3], s[4:5], 0x8                         // 000000001308: C00A0002 00000008
	v_lshlrev_b32_e32 v0, 2, v0                                // 000000001310: 24000082
	v_lshl_or_b32 v0, s6, 9, v0                                // 000000001314: D2000000 04011206
	v_ashrrev_i32_e32 v1, 31, v0                               // 00000000131C: 2202009F
	s_waitcnt lgkmcnt(0)                                       // 000000001320: BF8CC07F
	v_cmp_gt_i32_e32 vcc, s7, v0                               // 000000001324: 7D880007
	s_and_saveexec_b64 s[4:5], vcc                             // 000000001328: BE84206A
	s_cbranch_execz 14                                         // 00000000132C: BF88000E <zluda_vec4_sq_float+0x68>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 000000001330: D28F0002 00020082
	v_mov_b32_e32 v5, s3                                       // 000000001338: 7E0A0203
	v_add_co_u32_e32 v4, vcc, s2, v2                           // 00000000133C: 32080402
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 000000001340: 380A0705
	global_load_dword v4, v[4:5], off                          // 000000001344: DC508000 047F0004
	v_mov_b32_e32 v5, s1                                       // 00000000134C: 7E0A0201
	v_add_co_u32_e32 v2, vcc, s0, v2                           // 000000001350: 32040400
	v_addc_co_u32_e32 v3, vcc, v5, v3, vcc                     // 000000001354: 38060705
	s_waitcnt vmcnt(0)                                         // 000000001358: BF8C0F70
	v_mul_f32_e32 v4, v4, v4                                   // 00000000135C: 0A080904
	global_store_dword v[2:3], v4, off                         // 000000001360: DC708000 007F0402
	s_or_b64 exec, exec, s[4:5]                                // 000000001368: 87FE047E
	v_or_b32_e32 v2, 1, v0                                     // 00000000136C: 28040081
	v_cmp_gt_i32_e32 vcc, s7, v2                               // 000000001370: 7D880407
	s_and_saveexec_b64 s[4:5], vcc                             // 000000001374: BE84206A
	s_cbranch_execz 14                                         // 000000001378: BF88000E <zluda_vec4_sq_float+0xb4>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 00000000137C: D28F0002 00020082
	v_mov_b32_e32 v5, s3                                       // 000000001384: 7E0A0203
	v_add_co_u32_e32 v4, vcc, s2, v2                           // 000000001388: 32080402
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 00000000138C: 380A0705
	global_load_dword v4, v[4:5], off offset:4                 // 000000001390: DC508004 047F0004
	v_mov_b32_e32 v5, s1                                       // 000000001398: 7E0A0201
	v_add_co_u32_e32 v2, vcc, s0, v2                           // 00000000139C: 32040400
	v_addc_co_u32_e32 v3, vcc, v5, v3, vcc                     // 0000000013A0: 38060705
	s_waitcnt vmcnt(0)                                         // 0000000013A4: BF8C0F70
	v_mul_f32_e32 v4, v4, v4                                   // 0000000013A8: 0A080904
	global_store_dword v[2:3], v4, off offset:4                // 0000000013AC: DC708004 007F0402
	s_or_b64 exec, exec, s[4:5]                                // 0000000013B4: 87FE047E
	v_or_b32_e32 v2, 2, v0                                     // 0000000013B8: 28040082
	v_cmp_gt_i32_e32 vcc, s7, v2                               // 0000000013BC: 7D880407
	s_and_saveexec_b64 s[4:5], vcc                             // 0000000013C0: BE84206A
	s_cbranch_execz 14                                         // 0000000013C4: BF88000E <zluda_vec4_sq_float+0x100>
	v_lshlrev_b64 v[2:3], 2, v[0:1]                            // 0000000013C8: D28F0002 00020082
	v_mov_b32_e32 v5, s3                                       // 0000000013D0: 7E0A0203
	v_add_co_u32_e32 v4, vcc, s2, v2                           // 0000000013D4: 32080402
	v_addc_co_u32_e32 v5, vcc, v5, v3, vcc                     // 0000000013D8: 380A0705
	global_load_dword v4, v[4:5], off offset:8                 // 0000000013DC: DC508008 047F0004
	v_mov_b32_e32 v5, s1                                       // 0000000013E4: 7E0A0201
	v_add_co_u32_e32 v2, vcc, s0, v2                           // 0000000013E8: 32040400
	v_addc_co_u32_e32 v3, vcc, v5, v3, vcc                     // 0000000013EC: 38060705
	s_waitcnt vmcnt(0)                                         // 0000000013F0: BF8C0F70
	v_mul_f32_e32 v4, v4, v4                                   // 0000000013F4: 0A080904
	global_store_dword v[2:3], v4, off offset:8                // 0000000013F8: DC708008 007F0402
	s_or_b64 exec, exec, s[4:5]                                // 000000001400: 87FE047E
	v_or_b32_e32 v2, 3, v0                                     // 000000001404: 28040083
	v_cmp_gt_i32_e32 vcc, s7, v2                               // 000000001408: 7D880407
	s_and_saveexec_b64 s[4:5], vcc                             // 00000000140C: BE84206A
	s_cbranch_execz 14                                         // 000000001410: BF88000E <zluda_vec4_sq_float+0x14c>
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000001414: D28F0000 00020082
	v_mov_b32_e32 v3, s3                                       // 00000000141C: 7E060203
	v_add_co_u32_e32 v2, vcc, s2, v0                           // 000000001420: 32040002
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc                     // 000000001424: 38060303
	global_load_dword v2, v[2:3], off offset:12                // 000000001428: DC50800C 027F0002
	v_mov_b32_e32 v3, s1                                       // 000000001430: 7E060201
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 000000001434: 32000000
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 000000001438: 38020303
	s_waitcnt vmcnt(0)                                         // 00000000143C: BF8C0F70
	v_mul_f32_e32 v2, v2, v2                                   // 000000001440: 0A040502
	global_store_dword v[0:1], v2, off offset:12               // 000000001444: DC70800C 007F0200
	s_endpgm                                                   // 00000000144C: BF810000
