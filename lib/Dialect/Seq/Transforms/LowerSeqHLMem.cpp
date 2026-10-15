//===- LowerSeqHLMem.cpp - seq.hlmem lowering -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass pattern matches lowering patterns on seq.hlmem ops and referencing
// ports.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_LOWERSEQHLMEM
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;

namespace {

struct SimpleBehavioralMemoryLowering
    : public OpConversionPattern<seq::HLMemOp> {
  // A simple behavioral SV implementation of a HLMemOp. This is intended as a
  // fall-back pattern if any other higher benefit/target-specific patterns
  // failed to match.
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::HLMemOp mem, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Only support unidimensional memories.
    auto memType = mem.getMemType();
    if (memType.getShape().size() != 1)
      return rewriter.notifyMatchFailure(
          mem, "only unidimensional memories are supported");
    auto size = memType.getShape()[0];

    // Gather up the referencing ops.
    llvm::SmallVector<seq::ReadPortOp> readOps;
    llvm::SmallVector<seq::WritePortOp> writeOps;
    for (auto *user : mem.getHandle().getUsers()) {
      auto res = llvm::TypeSwitch<Operation *, LogicalResult>(user)
                     .Case([&](seq::ReadPortOp op) {
                       readOps.push_back(op);
                       return success();
                     })
                     .Case([&](seq::WritePortOp op) {
                       writeOps.push_back(op);
                       return success();
                     })
                     .Default([&](Operation *op) { return failure(); });
      if (failed(res))
        return rewriter.notifyMatchFailure(user, "unsupported port type");
    }

    auto clk = mem.getClk();
    auto rst = mem.getRst();
    auto memName = mem.getName();

    // Create the SV memory. It carries the memory's SV attributes, such as
    // the RAM style a synthesis tool should implement it in.
    hw::UnpackedArrayType memArrType =
        hw::UnpackedArrayType::get(memType.getElementType(), size);
    auto svReg = sv::RegOp::create(rewriter, mem.getLoc(), memArrType,
                                   mem.getNameAttr());
    if (auto attrs = sv::getSVAttributes(mem))
      sv::setSVAttributes(svReg, attrs);
    Value svMem = svReg.getResult();

    // Create write ports, each in its own always ff block. Synthesis tools
    // infer a multi-port RAM only from one write port per process: Vivado
    // turns a memory written by two ports in the same process into
    // registers. Two ports writing one address in the same cycle collide:
    // a block RAM keeps neither write reliably, and the order of the blocks
    // in simulation is unspecified.
    for (auto writeOp : writeOps)
      if (writeOp.getLatency() != 1)
        return rewriter.notifyMatchFailure(
            writeOp, "only supports write ports with latency == 1");

    struct Write {
      Location loc;
      Value addr;
      Value en;
    };
    llvm::SmallVector<Write> writes;
    auto hwClk = seq::FromClockOp::create(rewriter, clk.getLoc(), clk);
    for (auto writeOp : writeOps) {
      Location loc = writeOp.getLoc();
      Value addr = writeOp.getAddresses()[0];
      Value data = writeOp.getInData();
      Value en = writeOp.getWrEn();
      writes.push_back({loc, addr, en});
      sv::AlwaysFFOp::create(
          rewriter, loc, sv::EventControl::AtPosEdge, hwClk,
          sv::ResetType::SyncReset, sv::EventControl::AtPosEdge, rst, [&] {
            // Perform write upon write enable being high.
            sv::IfOp::create(rewriter, loc, en, [&] {
              Value memLoc =
                  sv::ArrayIndexInOutOp::create(rewriter, loc, svMem, addr);
              sv::PAssignOp::create(rewriter, loc, memLoc, data);
            });
          });
      rewriter.eraseOp(writeOp);
    }

    // Check in simulation that no two write ports write one address in the
    // same cycle, so a design never relies on either write winning. Writes
    // are off in reset, and so is the check.
    if (writes.size() > 1) {
      Location loc = mem.getLoc();
      Value collision;
      for (auto [i, a] : llvm::enumerate(writes)) {
        for (const Write &b : llvm::ArrayRef(writes).drop_front(i + 1)) {
          Value sameAddr = comb::ICmpOp::create(
              rewriter, loc, comb::ICmpPredicate::eq, a.addr, b.addr);
          Value pair = comb::AndOp::create(
              rewriter, loc, comb::AndOp::create(rewriter, loc, a.en, b.en),
              sameAddr);
          collision = collision
                          ? comb::OrOp::create(rewriter, loc, collision, pair)
                          : pair;
        }
      }
      verif::ClockedAssertOp::create(
          rewriter, loc, comb::createOrFoldNot(rewriter, loc, collision),
          verif::ClockEdge::Pos, hwClk,
          comb::createOrFoldNot(rewriter, loc, rst),
          rewriter.getStringAttr(memName + "_write_collision"));
    }

    // Create read ports. When latency > 1, we perform the read 1 cycle before
    // the latency deadline, and register the read output. By doing so, we avoid
    // start a critical path right after the combinational read.
    // TODO: When latency > 2, and assuming that the exact read time is flexible
    // within the latency window, we could decide on whether to buffer the read
    // data or the read address more, based on which is narrower (saving area).
    for (auto [ri, readOp] : llvm::enumerate(readOps)) {
      rewriter.setInsertionPointAfter(readOp);
      auto loc = readOp.getLoc();

      auto readAddress = readOp.getAddresses()[0];
      unsigned latency = readOp.getLatency();
      unsigned addressDelayCycles = latency - 1;
      if (latency > 0) {
        // Materialize any delays on the read address.
        for (unsigned i = 0; i < addressDelayCycles; ++i) {
          readAddress = seq::CompRegOp::create(
              rewriter, loc, readAddress, clk,
              rewriter.getStringAttr(memName + "_rdaddr" + std::to_string(ri) +
                                     "_dly" + std::to_string(i)));
        }
      }

      // Create a combinational read.
      Value memLoc =
          sv::ArrayIndexInOutOp::create(rewriter, loc, svMem, readAddress);
      Value readData = sv::ReadInOutOp::create(rewriter, loc, memLoc);
      if (latency > 0) {
        // Register the read data.
        readData = seq::CompRegOp::create(
            rewriter, loc, readData, clk,
            rewriter.getStringAttr(memName + "_rd" + std::to_string(ri) +
                                   "_reg"));
      }
      rewriter.replaceOp(readOp, {readData});
    }

    rewriter.eraseOp(mem);
    return success();
  }
};
struct LowerSeqHLMemPass
    : public circt::seq::impl::LowerSeqHLMemBase<LowerSeqHLMemPass> {
  void runOnOperation() override;
};

} // namespace

void LowerSeqHLMemPass::runOnOperation() {
  hw::HWModuleOp top = getOperation();

  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);

  // Lowering patterns must lower away all HLMem-related operations.
  target.addIllegalOp<seq::HLMemOp, seq::ReadPortOp, seq::WritePortOp>();
  target.addLegalDialect<sv::SVDialect, seq::SeqDialect, hw::HWDialect,
                         comb::CombDialect, verif::VerifDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<SimpleBehavioralMemoryLowering>(&ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}
