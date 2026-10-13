/*
 * Copyright © 2018 Intel Corporation
 * Copyright © 2021 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nir_test.h"

namespace {

class algebraic_test_base : public nir_test {
protected:
   algebraic_test_base();

   virtual void run_pass()=0;

   void test_op(nir_op op, nir_def *src0, nir_def *src1, nir_def *src2,
                nir_def *src3, const char *desc);

   void test_2src_op(nir_op op, int64_t src0, int64_t src1);

   void require_one_alu(nir_op op);

   nir_variable *res_var;
};

algebraic_test_base::algebraic_test_base()
   : nir_test::nir_test("nir_opt_algebraic_test")
{
   res_var = nir_local_variable_create(b->impl, glsl_int_type(), "res");
}

void algebraic_test_base::test_op(nir_op op, nir_def *src0, nir_def *src1,
                                     nir_def *src2, nir_def *src3, const char *desc)
{
   nir_def *res_deref = &nir_build_deref_var(b, res_var)->def;

   /* create optimized expression */
   nir_intrinsic_instr *optimized_instr = nir_build_store_deref(
      b, res_deref, nir_build_alu(b, op, src0, src1, src2, src3), 0x1);

   run_pass();
   b->cursor = nir_after_cf_list(&b->impl->body);

   /* create reference expression */
   nir_intrinsic_instr *ref_instr = nir_build_store_deref(
      b, res_deref, nir_build_alu(b, op, src0, src1, src2, src3), 0x1);

   /* test equality */
   nir_opt_constant_folding(b->shader);

   ASSERT_TRUE(nir_src_is_const(ref_instr->src[1]));
   ASSERT_TRUE(nir_src_is_const(optimized_instr->src[1]));

   uint64_t ref = nir_src_as_uint(ref_instr->src[1]);
   uint64_t optimized = nir_src_as_uint(optimized_instr->src[1]);

   EXPECT_EQ(ref, optimized) << "Test input: " << desc;

   /* reset shader */
   exec_list_make_empty(&nir_start_block(b->impl)->instr_list);
   b->cursor = nir_after_cf_list(&b->impl->body);
}

void algebraic_test_base::test_2src_op(nir_op op, int64_t src0, int64_t src1)
{
   char desc[128];
   snprintf(desc, sizeof(desc), "%s(%" PRId64 ", %" PRId64 ")", nir_op_infos[op].name, src0, src1);
   test_op(op, nir_imm_int(b, src0), nir_imm_int(b, src1), NULL, NULL, desc);
}

void algebraic_test_base::require_one_alu(nir_op op)
{
   unsigned count = 0;
   nir_foreach_instr(instr, nir_start_block(b->impl)) {
      if (instr->type == nir_instr_type_alu) {
         ASSERT_TRUE(nir_instr_as_alu(instr)->op == op);
         ASSERT_EQ(count, 0);
         count++;
      }
   }
}

class nir_opt_algebraic_test : public algebraic_test_base {
protected:
   virtual void run_pass() {
      nir_opt_algebraic(b->shader);
   }
};

class nir_opt_idiv_const_test : public algebraic_test_base {
protected:
   virtual void run_pass() {
      nir_opt_idiv_const(b->shader, 8);
   }
};

class nir_opt_mqsad_test : public algebraic_test_base {
protected:
   virtual void run_pass() {
      nir_opt_mqsad(b->shader);
   }
};

TEST_F(nir_opt_algebraic_test, umod_pow2_src2)
{
   for (int i = 0; i <= 9; i++)
      test_2src_op(nir_op_umod, i, 4);
   test_2src_op(nir_op_umod, UINT32_MAX, 4);
}

TEST_F(nir_opt_algebraic_test, imod_pow2_src2)
{
   for (int i = -9; i <= 9; i++) {
      test_2src_op(nir_op_imod, i, 4);
      test_2src_op(nir_op_imod, i, -4);
      test_2src_op(nir_op_imod, i, INT32_MIN);
   }
   test_2src_op(nir_op_imod, INT32_MAX, 4);
   test_2src_op(nir_op_imod, INT32_MAX, -4);
   test_2src_op(nir_op_imod, INT32_MIN, 4);
   test_2src_op(nir_op_imod, INT32_MIN, -4);
   test_2src_op(nir_op_imod, INT32_MIN, INT32_MIN);
}

TEST_F(nir_opt_algebraic_test, irem_pow2_src2)
{
   for (int i = -9; i <= 9; i++) {
      test_2src_op(nir_op_irem, i, 4);
      test_2src_op(nir_op_irem, i, -4);
   }
   test_2src_op(nir_op_irem, INT32_MAX, 4);
   test_2src_op(nir_op_irem, INT32_MAX, -4);
   test_2src_op(nir_op_irem, INT32_MIN, 4);
   test_2src_op(nir_op_irem, INT32_MIN, -4);
}

TEST_F(nir_opt_algebraic_test, idiv_pow2_signed_rounding)
{
   for (unsigned bits : {8, 16, 32, 64}) {
      res_var = nir_local_variable_create(b->impl, glsl_intN_t_type(bits), "res");
      const uint64_t sign_bit = UINT64_C(1) << (bits - 1);

      for (unsigned shift = 0; shift < bits; shift++) {
         const uint64_t magnitude = UINT64_C(1) << shift;
         for (bool negative : {false, true}) {
            if (!negative && magnitude == sign_bit)
               continue;
            const uint64_t divisor = negative ? -magnitude : magnitude;
            uint64_t random = UINT64_C(0x4d595df4d0f33173);
            const uint64_t edges[] = {
               0, 1, UINT64_MAX, sign_bit, sign_bit + 1, sign_bit - 1,
               magnitude - 1, magnitude, magnitude + 1,
               -magnitude - 1, -magnitude, -magnitude + 1,
            };
            for (unsigned i = 0; i < ARRAY_SIZE(edges) + 256; i++) {
               random = random * UINT64_C(6364136223846793005) + 1;
               const uint64_t dividend = i < ARRAY_SIZE(edges) ? edges[i] :
                  bits == 8 ? i - ARRAY_SIZE(edges) : random;
               /* Signed division overflow is undefined. Constant folding
                * chooses zero, while both lowerings can wrap to INT_MIN.
                */
               const uint64_t mask = UINT64_MAX >> (64 - bits);
               if (negative && shift == 0 && (dividend & mask) == sign_bit)
                  continue;
               char desc[128];
               snprintf(desc, sizeof(desc), "idiv%u(0x%" PRIx64 ", 0x%" PRIx64 ")",
                        bits, dividend, divisor);
               test_op(nir_op_idiv, nir_imm_intN_t(b, dividend, bits),
                       nir_imm_intN_t(b, divisor, bits), NULL, NULL, desc);
            }
         }
      }
   }
}

TEST_F(nir_opt_algebraic_test, msad)
{
   options.lower_bitfield_extract = true;
   options.has_bfe = true;
   options.has_msad = true;

   nir_def *src0 = nir_load_var(b, nir_local_variable_create(b->impl, glsl_int_type(), "src0"));
   nir_def *src1 = nir_load_var(b, nir_local_variable_create(b->impl, glsl_int_type(), "src1"));

   /* This mimics the sequence created by vkd3d-proton. */
   nir_def *res = NULL;
   for (unsigned i = 0; i < 4; i++) {
      nir_def *ref = nir_ubitfield_extract(b, src0, nir_imm_int(b, i * 8), nir_imm_int(b, 8));
      nir_def *src = nir_ubitfield_extract(b, src1, nir_imm_int(b, i * 8), nir_imm_int(b, 8));
      nir_def *is_ref_zero = nir_ieq_imm(b, ref, 0);
      nir_def *abs_diff = nir_iabs(b, nir_isub(b, ref, src));
      nir_def *masked_diff = nir_bcsel(b, is_ref_zero, nir_imm_int(b, 0), abs_diff);
      if (res)
         res = nir_iadd(b, res, masked_diff);
      else
         res = masked_diff;
   }

   nir_store_var(b, res_var, res, 0x1);

   while (nir_opt_algebraic(b->shader)) {
      nir_opt_constant_folding(b->shader);
      nir_opt_dce(b->shader);
   }

   require_one_alu(nir_op_msad_4x8);
}

TEST_F(nir_opt_mqsad_test, mqsad)
{
   options.lower_bitfield_extract = true;
   options.has_bfe = true;
   options.has_msad = true;
   options.has_shfr32 = true;

   nir_def *ref = nir_load_var(b, nir_local_variable_create(b->impl, glsl_int_type(), "ref"));
   nir_def *src = nir_load_var(b, nir_local_variable_create(b->impl, glsl_ivec_type(2), "src"));
   nir_def *accum = nir_load_var(b, nir_local_variable_create(b->impl, glsl_ivec_type(4), "accum"));

   nir_def *srcx = nir_channel(b, src, 0);
   nir_def *srcy = nir_channel(b, src, 1);

   nir_def *res[4];
   for (unsigned i = 0; i < 4; i++) {
      nir_def *src1 = srcx;
      switch (i) {
      case 0:
         break;
      case 1:
         src1 = nir_bitfield_select(b, nir_imm_int(b, 0xff000000), nir_ishl_imm(b, srcy, 24),
                                    nir_ushr_imm(b, srcx, 8));
         break;
      case 2:
         src1 = nir_bitfield_select(b, nir_imm_int(b, 0xffff0000), nir_ishl_imm(b, srcy, 16),
                                    nir_extract_u16(b, srcx, nir_imm_int(b, 1)));
         break;
      case 3:
         src1 = nir_bitfield_select(b, nir_imm_int(b, 0xffffff00), nir_ishl_imm(b, srcy, 8),
                                    nir_extract_u8_imm(b, srcx, 3));
         break;
      }

      res[i] = nir_msad_4x8(b, ref, src1, nir_channel(b, accum, i));
   }

   nir_store_var(b, nir_local_variable_create(b->impl, glsl_ivec_type(4), "res"), nir_vec(b, res, 4), 0xf);

   while (nir_opt_algebraic(b->shader)) {
      nir_opt_constant_folding(b->shader);
      nir_opt_dce(b->shader);
   }

   ASSERT_TRUE(nir_opt_mqsad(b->shader));
   nir_opt_copy_prop(b->shader);
   nir_opt_dce(b->shader);

   require_one_alu(nir_op_mqsad_4x8);
}

TEST_F(nir_opt_idiv_const_test, umod)
{
   for (uint32_t d : {16u, 17u, 0u, UINT32_MAX}) {
      for (int i = 0; i <= 40; i++)
         test_2src_op(nir_op_umod, i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_umod, UINT32_MAX - i, d);
   }
}

TEST_F(nir_opt_idiv_const_test, imod)
{
   for (int32_t d : {16, -16, 17, -17, 0, INT32_MIN, INT32_MAX}) {
      for (int i = -40; i <= 40; i++)
         test_2src_op(nir_op_imod, i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_imod, INT32_MIN + i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_imod, INT32_MAX - i, d);
   }
}

TEST_F(nir_opt_idiv_const_test, irem)
{
   for (int32_t d : {16, -16, 17, -17, 0, INT32_MIN, INT32_MAX}) {
      for (int i = -40; i <= 40; i++)
         test_2src_op(nir_op_irem, i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_irem, INT32_MIN + i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_irem, INT32_MAX - i, d);
   }
}

}
