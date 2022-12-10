// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is the JavaScript side of loop_emscripten.c
//
// References:
//   * https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html
//   * https://github.com/evanw/emscripten-library-generator
//   * https://github.com/emscripten-core/emscripten/tree/main/src

const LibraryLoopEmscripten = {
  $loop_emscripten_support__postset: 'loop_emscripten_support();',
  $loop_emscripten_support: function() {
    const IREE_STATUS_OK = 0;
    const IREE_STATUS_CODE_MASK = 0x1F;
    const IREE_STATUS_ABORTED = 10 & IREE_STATUS_CODE_MASK;
    const IREE_STATUS_OUT_OF_RANGE = 11 & IREE_STATUS_CODE_MASK;

    class LoopOperation {
      abort() {}
    }

    // IREE_LOOP_COMMAND_CALL
    class LoopOperationCall extends LoopOperation {
      constructor(scope, operationId, callback, user_data, loop) {
        super();

        this.callback = callback;
        this.user_data = user_data;
        this.loop = loop;

        this.timeoutId = setTimeout(() => {
          Module['dynCall'](
              'iiii', this.callback, this.user_data, this.loop, IREE_STATUS_OK);
          // TODO(scotttodd): handle the returned status (sticky failure state?)
          delete scope.pendingOperations[operationId];
        }, 0);
      }

      abort() {
        clearTimeout(this.timeoutId);
        Module['dynCall'](
            'iiii', this.callback, this.user_data, this.loop,
            IREE_STATUS_ABORTED);
      }
    }

    class LoopEmscriptenScope {
      constructor() {
        this.nextOperationId = 0;

        // Dictionary of operationIds -> LoopOperations.
        this.pendingOperations = {};
      }

      destroy() {
        for (const [id, operation] of Object.entries(this.pendingOperations)) {
          operation.abort();
          delete this.pendingOperations[id];
        }
      }

      command_call(callback, user_data, loop) {
        // TODO(scotttodd): assert not destroyed to avoid reentrant queueing?
        const operationId = this.nextOperationId++;
        this.pendingOperations[operationId] =
            new LoopOperationCall(this, operationId, callback, user_data, loop);
        return IREE_STATUS_OK;
      }
    }

    class LoopEmscripten {
      constructor() {
        this.nextScopeHandle = 0;

        // Dictionary of scopeHandles -> LoopEmscriptenScopes.
        this.scopes = {};
      }

      loop_allocate_scope() {
        const scopeHandle = this.nextScopeHandle++;
        this.scopes[scopeHandle] = new LoopEmscriptenScope();
        return scopeHandle;
      }

      loop_free_scope(scope_handle) {
        if (!(scope_handle in this.scopes)) return;

        const scope = this.scopes[scope_handle];
        scope.destroy();
        delete this.scopes[scope_handle];
      }

      loop_command_call(scope_handle, callback, user_data, loop) {
        if (!(scope_handle in this.scopes)) return IREE_STATUS_OUT_OF_RANGE;

        const scope = this.scopes[scope_handle];
        return scope.command_call(callback, user_data, loop);
      }
    }

    const instance = new LoopEmscripten();
    _loop_allocate_scope = instance.loop_allocate_scope.bind(instance);
    _loop_free_scope = instance.loop_free_scope.bind(instance);
    _loop_command_call = instance.loop_command_call.bind(instance);
  },

  loop_allocate_scope: function() {},
  loop_allocate_scope__deps: ['$loop_emscripten_support'],
  loop_free_scope: function() {},
  loop_free_scope__deps: ['$loop_emscripten_support'],
  loop_command_call: function() {},
  loop_command_call__deps: ['$loop_emscripten_support'],
}

mergeInto(LibraryManager.library, LibraryLoopEmscripten);
