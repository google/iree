// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_GLOBALOPTIMIZATION_DATALAYOUTUTILS_H_
#define IREE_GLOBALOPTIMIZATION_DATALAYOUTUTILS_H_

#include <optional>
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Location;
class OpBuilder;
class Operation;
class RewriterBase;
class Value;
} // namespace mlir

namespace mlir::iree_compiler::GlobalOptimization {

/// UNINITIALIZED: This node has not been initialized.
/// INTERMEDIATE: It is possible to propagate a layout through this node.
/// BARRIER: It is not possible to propagate a layout through this node.
enum class DataLayoutNodeType {
  UNINITIALIZED,
  INTERMEDIATE,
  BARRIER,
};

/// TODO: Abstractify DataLayoutTransformation to decouple from specific types
/// of transformations.
class DataLayoutTransformation {
public:
  DataLayoutTransformation(ShapedType orig, ShapedType transformed)
      : originalType(orig), transformedType(transformed){};
  DataLayoutTransformation(ShapedType orig) : originalType(orig){};
  DataLayoutTransformation(DataLayoutTransformation &other) {
    originalType = other.originalType;
    transformedType = other.transformedType;
    innerDimsPos = other.innerDimsPos;
    innerTileSizes = other.innerTileSizes;
    outerDimsPerm = other.outerDimsPerm;
    correspondingTransformedIndices = other.correspondingTransformedIndices;
  };
  DataLayoutTransformation(){};

  ShapedType getOriginalType() const { return originalType; };
  ShapedType getTransformedType() const { return transformedType; };
  SmallVector<int64_t> getInnerDimsPos() const { return innerDimsPos; };
  SmallVector<int64_t> getInnerTileSizes() const { return innerTileSizes; };
  SmallVector<int64_t> getOuterDimsPerm() const { return outerDimsPerm; };
  std::optional<TypedAttr> getConstantPadValue() const {
    return constantPadValue;
  };
  SmallVector<int64_t> getCorrespondingTransformedIndices() const {
    return correspondingTransformedIndices;
  };
  void setOriginalType(ShapedType type) { originalType = type; };
  void setTransformedType(ShapedType type) { transformedType = type; };
  void setInnerDimsPos(SmallVector<int64_t> pos) { innerDimsPos = pos; };
  void setInnerTileSizes(SmallVector<int64_t> tiles) {
    innerTileSizes = tiles;
  };
  void setOuterDimsPerm(SmallVector<int64_t> perm) { outerDimsPerm = perm; };
  void setConstantPadValue(std::optional<TypedAttr> attr) {
    constantPadValue = attr;
  };
  void setCorrespondingTransformedIndices(SmallVector<int64_t> inds) {
    correspondingTransformedIndices = inds;
  };

  /// Get the transformed layout at the `newValue`, given the `currentValue`
  /// with `this` layout. Return true for a successful transformation, and
  /// return false if the transformation is not supported.
  bool transformLayout(Value currentValue, Value newValue);

  /// Combine the information from this transform with another transform, and
  /// return whether or not information was gained.
  bool combineLayout(DataLayoutTransformation other);

  /// Return whether this transform is valid. For now, only check that there is
  /// an originalType and a transformedType.
  const bool hasValidTransform();

  /// Return true if the transformed indices in this transformation overlap with
  /// the transformed indices of the other transformation.
  bool isIntersecting(DataLayoutTransformation other);

  /// Create an ArrayAttr containing transformation information for debugging.
  ArrayAttr makeTransformArrayAttr(MLIRContext *ctx);

  /// Return a new identity transformation.
  static DataLayoutTransformation *getIdentityTransformation(ShapedType type) {
    auto *tf = new DataLayoutTransformation(type, type);
    tf->setCorrespondingTransformedIndices(
        llvm::to_vector(llvm::seq<int64_t>(0, type.getRank())));
    return tf;
  }

private:
  /// Transform the layout as if propagating through an operation, from the
  /// `currentValue` to `newValue`, and return the new layout.
  bool transform(Operation *op, Value currentValue, Value newValue);

  /// The original type corresponding to the source of this layout.
  ShapedType originalType;
  /// The type of the layout source after the transformation is applied.
  ShapedType transformedType;
  /// Transformation metadate from `originalType`->`transformedType, represented
  /// as pack metadata (innerDimsPos, innerTileSizes, outerDimsPerm) for now.
  SmallVector<int64_t> innerDimsPos;
  SmallVector<int64_t> innerTileSizes;
  SmallVector<int64_t> outerDimsPerm;
  /// Optional padding value for packed layouts.
  std::optional<TypedAttr> constantPadValue = std::nullopt;
  /// Indices in the `originalType` corresponding to each index of the value
  /// associated with this DataLayoutTransformation.
  SmallVector<int64_t> correspondingTransformedIndices;
};

//===----------------------------------------------------------------------===//
// Analysis helpers
//===----------------------------------------------------------------------===//

/// Analyze the producer and users of this value, and return the node type for
/// the given value.
DataLayoutNodeType getNodeTypeForValue(Value value);

/// If the node is a terminal node (e.g., the source of a layout, like a value
/// next to a global load or store), then return the layoutIDs corresponding to
/// that terminal node.
SmallVector<StringRef> getTerminalNodeIDs(Value value);

//===----------------------------------------------------------------------===//
// Pass helpers
//===----------------------------------------------------------------------===//

/// Rewrite a Util::GlobalOp into the new layout indicated by the given
/// DataLayoutTransformation, and rewrite all GlobalLoadOps or GlobalStoreOps
/// of the `global`, passed in `edgeNodes` as the results or inputs to the
/// load/store ops. Also, create an initializer to fill the new packed global
/// with the padding value of the pack in the `transform`.
LogicalResult transformGlobalsToNewLayout(IRRewriter &rewriter,
                                          SmallVector<Value> edgeNodes,
                                          DataLayoutTransformation *transform,
                                          IREE::Util::GlobalOp global,
                                          SymbolTable moduleSymbols);

//===----------------------------------------------------------------------===//
// Attribute helpers
//===----------------------------------------------------------------------===//

/// Padding values can get in the way of unpack(pack(x)) foldings, so this sets
/// a `__foldable_pack_unpack__` attribute, which indicates that the op came
/// from a given data layout, and can be folded with the inverse of the op as
/// long as it also has the same attribute.
void setFoldablePackUnPackAttribute(Operation *op);

/// Return whether the op has the `__foldable_pack_unpack__` attribute.
bool hasFoldablePackUnPackAttribute(Operation *op);

/// Get the DataLayoutNodeType of an annotated op.
std::optional<DataLayoutNodeType> getNodeTypeFromAttr(Operation *op);

/// Set the `__node_type__` attribute for the op.
void setNodeTypeAttribute(Operation *op, DataLayoutNodeType nodeType);

/// Annotate an op for debugging with the layoutID and original + transformed
/// type for a transformation.
void setDataLayoutTransformationAttributes(Operation *op,
                                           DataLayoutTransformation *transform,
                                           StringRef transformID);

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_GLOBALOPTIMIZATION_DATALAYOUTUTILS_H_
