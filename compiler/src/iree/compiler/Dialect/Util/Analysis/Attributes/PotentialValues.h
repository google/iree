// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_ATTRIBUTES_POTENTIAL_VALUES_H_
#define IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_ATTRIBUTES_POTENTIAL_VALUES_H_

#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

namespace mlir::iree_compiler::IREE::Util {

//===----------------------------------------------------------------------===//
// ConstantAttributePVS
// Potential value set of Attribute instances representing constant values.
// The yielded set of potential values will consist of |mlir::Attribute|
// instances representing the values of any constants or immutable globals
// directly addressed from the query position or reachable via
// control-flow without alteration. If the program flow passes any
// unrecognized operations, the set will include undef, indicating that
// there may be non-constants in the data-flow.
//===----------------------------------------------------------------------===//

class ConstantAttributePVS
    : public DFX::StateWrapper<DFX::PotentialValuesState<Attribute>,
                               DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<DFX::PotentialValuesState<Attribute>,
                                     DFX::ValueElement>;
  using BaseType::BaseType;

  static ConstantAttributePVS &createForPosition(const Position &pos,
                                                 DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ConstantAttributePVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "ConstantAttributePVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override;

private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
};

//===----------------------------------------------------------------------===//
// GlobalPVS
// Potential value set of global operations.
//===----------------------------------------------------------------------===//

class GlobalPVS : public DFX::StateWrapper<
                      DFX::PotentialConstantIntValuesState,
                      DFX::TypedOperationElement<IREE::Util::GlobalOp>> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialConstantIntValuesState,
                        DFX::TypedOperationElement<IREE::Util::GlobalOp>>;

  static GlobalPVS &createForPosition(const Position &pos,
                                      DFX::Solver &solver) {
    return *(new (solver.getAllocator()) GlobalPVS(pos));
  }

  const std::string getName() const override { return "GlobalPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override;

private:
  explicit GlobalPVS(const Position &pos) : BaseType(pos) {}

  void initializeOperation(IREE::Util::GlobalOp globalOp,
                           DFX::Solver &solver) override;
  ChangeStatus updateOperation(IREE::Util::GlobalOp globalOp,
                               DFX::Solver &solver) override;

  friend class DFX::Solver;
};

//===----------------------------------------------------------------------===//
// IntValuePVS
// Potential value set of global operations.
// TODO: This is really ConstantIntPVS. Rename.
//===----------------------------------------------------------------------===//

class IntValuePVS
    : public DFX::StateWrapper<DFX::PotentialConstantIntValuesState,
                               DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<DFX::PotentialConstantIntValuesState,
                                     DFX::ValueElement>;

  static IntValuePVS &createForPosition(const Position &pos,
                                        DFX::Solver &solver) {
    return *(new (solver.getAllocator()) IntValuePVS(pos));
  }

  const std::string getName() const override { return "IntValuePVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override;

private:
  explicit IntValuePVS(const Position &pos) : BaseType(pos) {}

  void initializeValue(Value value, DFX::Solver &solver) override;

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
  friend class DFX::Solver;
};

} // namespace mlir::iree_compiler::IREE::Util

#endif // IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_ATTRIBUTES_POTENTIAL_VALUES_H_
