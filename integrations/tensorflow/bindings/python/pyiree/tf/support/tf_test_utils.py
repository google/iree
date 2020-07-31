# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test utilities interop with TensorFlow."""

# pylint: disable=missing-docstring
# pylint: disable=protected-access
# pylint: disable=unsupported-assignment-operation

# This file uses the following abbreviations:
#   ref: reference – for the reference CompiledModule
#   tar: target - for one of the target CompiledModules

import collections
import copy
import inspect
import os
import re
import sys
import tempfile

from absl import flags
from absl import logging
import numpy as np
from pyiree.tf import compiler
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

flags.DEFINE_string("target_backends", None,
                    "Explicit comma-delimited list of target backends.")
flags.DEFINE_string(
    "debug_dir", None,
    "Specifies a directory to dump debug artifacts to. Defaults to "
    "--test_tmpdir")
flags.DEFINE_string(
    "artifacts_dir", None,
    "Specifies a directory to dump compilation artifacts and traces to. "
    "Defaults to the OS's tempdir.")
flags.DEFINE_string("reference_backend", "tf",
                    "The backend to treat as a source of truth.")
flags.DEFINE_bool(
    "summarize", True,
    "Summarize the inputs and outputs of each module trace logged to disk.")
FLAGS = flags.FLAGS
NUMPY_LINEWIDTH = 120


def _setup_test_debug_dir(test_name):
  global global_debug_dir

  # Use test_tempdir (which defaults to '/tmp/absl_testing/') if FLAGS.debug_dir
  # is not provided.
  parent = FLAGS.debug_dir if FLAGS.debug_dir is not None else FLAGS.test_tmpdir
  global_debug_dir = os.path.join(parent, test_name)

  # Create the directory.
  try:
    os.makedirs(global_debug_dir)
  except IOError:
    logging.exception("Error creating debug dir for: %s", global_debug_dir)


def _setup_artifacts_dir(module_name):
  parent_dir = FLAGS.artifacts_dir
  if parent_dir is None:
    parent_dir = os.path.join(tempfile.gettempdir(), "iree", "modules")
  artifacts_dir = os.path.join(parent_dir, module_name)
  logging.info("Saving compilation artifacts and traces to '%s'", artifacts_dir)

  # If the artifacts already exist then we overwrite/update them.
  try:
    # Use try/except instead of os.path.exists to address a race condition
    # between multiple tests targets.
    os.makedirs(artifacts_dir)
  except IOError:
    pass
  return artifacts_dir


class _VirtualModuleInstance(object):
  """Wraps a namedtuple of modules and represents a union of them."""

  def __init__(self, named_modules, match_spec):
    self._named_modules = named_modules
    self._match_spec = match_spec

  def __getattr__(self, attr):
    match_modules = {
        k: v
        for k, v in self._named_modules.items()
        if re.search(self._match_spec, k)
    }
    if not match_modules:
      raise AttributeError(
          "Module match spec '%s' did not match anything. (Have %r)" %
          (self._match_spec, self._named_modules.keys()))
    # Resolve functions on each.
    match_functions = {}
    for backend, module in match_modules.items():
      try:
        match_functions[backend] = getattr(module, attr)
      except:
        raise AttributeError(
            "Could not resolve function '%s' on backend module '%s'" %
            (attr, backend))
    return _VirtualFunctionWrapper(match_functions)


class _VirtualFunctionWrapper(object):
  """Wrapper around a virtual dict of functions."""

  def __init__(self, backend_function_dict):
    self._backend_function_dict = backend_function_dict

  def __call__(self, *args, **kwargs):
    all_results = {
        backend: f(*args, **kwargs)
        for backend, f in self._backend_function_dict.items()
    }
    # Turn it into a named tuple so we get nice class-like access to it.
    results_tuple_class = collections.namedtuple("Results", all_results.keys())
    return _make_multi_result_class(results_tuple_class)(*all_results.values())


def _recursive_check_same(result_ref, result_tgt, rtol=1e-6, atol=1e-6):
  same = True
  if not isinstance(result_tgt, type(result_ref)):
    raise ValueError("Types of the outputs must be the same, but have '{}' and "
                     "'{}'".format(type(result_ref), type(result_tgt)))
  if isinstance(result_ref, dict):
    if result_ref.keys() != result_tgt.keys():
      raise ValueError("Outputs must have the same structure, but have '{}' and"
                       " '{}'".format(result_ref.keys(), result_tgt.keys()))
    for key in result_ref.keys():
      same = same and _recursive_check_same(result_ref[key], result_tgt[key],
                                            rtol, atol)
      if not same:
        return False  # no need to go further they are different
  elif isinstance(result_ref, list):
    if len(result_ref) != len(result_tgt):
      raise ValueError("Outputs must have the same structure, but have '{}' and"
                       " '{}'".format(result_ref, result_tgt))
    for i in range(len(result_ref)):
      same = same and _recursive_check_same(result_ref[i], result_tgt[i], rtol,
                                            atol)
      if not same:
        return False  # no need to go further they are different
  elif isinstance(result_ref, np.ndarray):
    if isinstance(result_ref.flat[0], np.floating):
      return np.allclose(result_ref, result_tgt, rtol=rtol, atol=atol)
    else:
      return np.array_equal(result_ref, result_tgt)
  else:
    # this one need more checks
    return result_ref == result_tgt
  return same


def _collect_disagreements_recursively(mr, rtol=1e-6, atol=1e-6):
  """Compare result structs recursively and search for disagreements.

  Args:
    mr: A MultiResults namedtuple where each entry corresponds to a backend set
      of results.
    rtol: The relative tolerance parameter.
    atol: The absolute tolerance parameter.

  Returns:
    An equivalent MultiResults where each entry is an array of result names
    that disagree.
  """
  has_disagreement = False
  disagreement_list = [list() for _ in mr]
  for i in range(len(mr)):
    result_ref = mr[i]
    for j in range(len(mr)):
      if i < j:
        continue  # Don't check self and reverse comparisons
      result_tgt = mr[j]
      if not _recursive_check_same(result_ref, result_tgt, rtol, atol):
        has_disagreement = True
        disagreement_list[i].append(mr._fields[j])
  disagreements_tuple = collections.namedtuple("Disagreements", mr._fields)
  return has_disagreement, disagreements_tuple(*disagreement_list)


def _collect_disagreements(mr, predicate):
  """Verifies that result structs.

  Args:
    mr: A MultiResults namedtuple where each entry corresponds to a backend set
      of results.
    predicate: A predicate function which takes (a, b) and returns whether they
      should be considered equivalent.

  Returns:
    An equivalent MultiResults where each entry is an array of result names
    that disagree.
  """
  has_disagreement = False
  disagreement_list = [list() for _ in mr]
  for i in range(len(mr)):
    result_ref = mr[i]
    for j in range(len(mr)):
      if i == j:
        continue  # Don't check self.
      result_tgt = mr[j]
      if not predicate(result_ref, result_tgt):
        has_disagreement = True
        disagreement_list[i].append(mr._fields[j])
  disagreements_tuple = collections.namedtuple("Disagreements", mr._fields)
  return has_disagreement, disagreements_tuple(*disagreement_list)


def _make_multi_result_class(named_tuple_class):
  """Makes a class that wraps a mapping of backend results."""

  class MultiResults(named_tuple_class):
    """Wraps a mapping of results."""

    def assert_all_close(self, rtol=1e-6, atol=1e-6):
      predicate = (lambda a, b: np.allclose(a, b, rtol=rtol, atol=atol))
      has_disagreement, disagreements = _collect_disagreements(self, predicate)
      assert not has_disagreement, ("Multiple backends disagree (%r):\n%r" %
                                    (disagreements, self))
      return self

    def assert_all_equal(self):
      predicate = np.array_equal
      has_disagreement, disagreements = _collect_disagreements(self, predicate)
      assert not has_disagreement, ("Multiple backends disagree (%r):\n%r" %
                                    (disagreements, self))
      return self

    def assert_all_close_and_equal(self, rtol=1e-6, atol=1e-6):
      # it is a special case when output can be a nestet map of dict(), list()
      # with different types: int, float, string
      # in this case int and string must be equal and for float we use rtol,atol
      has_disagreement, disagreements = _collect_disagreements_recursively(
          self, rtol, atol)
      assert not has_disagreement, ("Multiple backends disagree (%r):\n%r" %
                                    (disagreements, self))
      return self

    def print(self):
      print(self)
      return self

    def save(self):
      for i in range(len(self)):
        result = self[i]  # output generated by a model
        field = self._fields[i]  # backend name
        fname = os.path.join(global_debug_dir, "output_{}".format(field))
        with open(fname, "w") as file:
          # content of txt file can be converted to py objects by eval(txt)
          file.write(str(result))
      return self

  return MultiResults


def _instantiate_backends(compiled_backends):
  """Creates a VirtualBackend namedtuple class for a dict.

  Args:
    compiled_backends: Dictionary of backend_name:ModuleInstance.

  Returns:
    a VirtualBackendsClass instance. The VirtualBackendsClass is a dynamically
    generated namedtuple mapping backend_name:ModuleInstance, where the
    ModuleInstance allows attribute resolution of public functions on the
    module. The VirtualBackendsClass also contributes some convenience methods
    for selecting all or a subset of matching backend modules.
  """
  tuple_class = collections.namedtuple("VirtualBackendsTuple",
                                       compiled_backends.keys())

  class VirtualBackendsClass(tuple_class):
    """Adds a __call__ method that creates a virtual module."""

    def multi(self, match_spec="."):
      """Selects multiple backends that match a regular expression."""
      return _VirtualModuleInstance(self._asdict(), match_spec)

    @property
    def all(self):
      """Shorthand for multi() which selects all backends."""
      return self.multi()

  reinitialized_modules = [
      module.create_reinitialized() for module in compiled_backends.values()
  ]
  return VirtualBackendsClass(*reinitialized_modules)


def compile_module(module_class, exported_names=()):
  """CompiledModuleTestCase decorator that compiles a tf.Module.

  A CompiledModule is created for each backend in --target_backends. They can
  be accessed individually via self.compiled_modules.backend_name or as a union
  via self.get_module().

  Args:
    module_class: the tf.Module subclass to compile.
    exported_names: optional iterable of strings representing which of
      module_class's functions to compile. If exported_names is empty all
      functions will be compiled.

  Returns:
    Class decorator function.
  """

  def decorator(cls):
    """Decorator Function."""
    if not issubclass(cls, CompiledModuleTestCase):
      logging.exception(
          "The 'compile_module' decorator must be applied to a "
          "CompiledModuleTestCase derived class, which %s is not.", cls)
    cls._module_class = module_class
    cls._exported_names = exported_names
    return cls

  return decorator


def _parse_target_backends(target_backends):
  """Decodes a comma-delimited string of backends into BackendInfo objects."""
  backends = []
  for backend_name in target_backends.split(","):
    if backend_name not in tf_utils.BackendInfo.ALL.keys():
      raise ValueError(
          "Invalid backend specification string '{}', unexpected name '{}';"
          " valid names are '{}'".format(target_backends, backend_name,
                                         tf_utils.BackendInfo.ALL.keys()))
    backends.append(tf_utils.BackendInfo.ALL[backend_name])
  return backends


def get_backends():
  """Gets the BackendInfo instances to test.

  By default all backends in BackendInfo will be used. Specific backends to
  run on can be specified using the `--target_backends` flag. If only "tf" is
  provided then it will be compared against itself.

  Returns:
    Sequence of BackendInfo that should be used.
  """
  if FLAGS.target_backends is not None:
    logging.info("Using backends from command line: %s", FLAGS.target_backends)
    backends = _parse_target_backends(FLAGS.target_backends)
    # If tf is the only backend then we will test it itself by adding tf_also.
    if len(backends) == 1 and "tf" == backends[0].name:
      backends.append(tf_utils.BackendInfo.ALL["tf_also"])
  else:
    # If no backends are specified, use them all.
    backends = list(tf_utils.BackendInfo.ALL.values())
  return backends


class CompiledModuleTestCase(tf.test.TestCase):
  """Compiles a tf.Module to multiple backends to test their correctness."""

  # Will be initialized by the @compile_module decorator.
  _module_class = None
  _exported_names = ()

  # Will be initialized in setUpClass to a dict of
  # {backend_name: CompiledModule}.
  _compiled_backends_dict = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls._module_class is None:
      raise AttributeError(
          "setUpClass was called but no module was specified. Specify a module "
          "to compile via the @tf_test_utils.compile_module decorator.")

    # Setup the debug directory for this test. Creates a global variable
    # `global_debug_dir`.
    _setup_test_debug_dir(test_name=cls.__name__)

    # Setup crash reproducer for the test.
    crash_reproducer_path = os.path.join(global_debug_dir, "reproducer.mlir")
    compiler.Context.default_crash_reproducer_path = crash_reproducer_path

    # Create a CompiledModule for each backend.
    try:
      backends = get_backends()
      cls._compiled_backends_dict = {}
      for backend_info in backends:
        compiled_backend = backend_info.CompiledModule(cls._module_class,
                                                       backend_info,
                                                       cls._exported_names,
                                                       global_debug_dir)
        cls._compiled_backends_dict[backend_info.name] = compiled_backend
    finally:
      # Disable crash reproducer (to avoid inadvertently overwriting this
      # path on a subsequent interaction).
      compiler.Context.default_crash_reproducer_path = None

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    self.compiled_modules = _instantiate_backends(self._compiled_backends_dict)

  def get_module(self):
    return self.compiled_modules.all


def get_target_backends():
  """Gets the BackendInfo instances to compare with the reference backend.

  By default all backends in BackendInfo will be used. Specific backends to
  run on can be specified using the `--target_backends` flag.

  Returns:
    Sequence of BackendInfo that should be used.
  """
  if FLAGS.target_backends is not None:
    logging.info("Using backends from command line: %s", FLAGS.target_backends)
    backends = _parse_target_backends(FLAGS.target_backends)
  else:
    # If no backends are specified, use them all.
    backends = list(tf_utils.BackendInfo.ALL.values())
  return backends


def _indent(input_str, indentation=2):
  """Indents a string by the specified number of spaces, defaulting to 2."""
  spaces = " " * indentation
  lines = input_str.split("\n")
  # Prepend spaces to each non-empty line.
  lines = [f"{spaces}{line}" if len(line) else line for line in lines]
  return "\n".join(lines)


class ModuleCall:

  def __init__(self, method_name, inputs, outputs, rtol=1e-6, atol=1e-6):
    """Records the details of a call to a CompiledModule."""
    self.method = method_name

    # Deepcopy to safegard against mutation.
    self.inputs = copy.deepcopy(inputs)
    if outputs is not None:
      outputs = copy.deepcopy(outputs)
    else:
      outputs = tuple()
    self.outputs = outputs if isinstance(outputs, tuple) else (outputs,)

    self.rtol = rtol
    self.atol = atol

  def get_tolerances(self):
    """Gets the floating point tolerances associated with this call."""
    return self.rtol, self.atol

  def __str__(self):
    prior_printoptions = np.get_printoptions()
    np.set_printoptions(linewidth=NUMPY_LINEWIDTH)

    header = f"Method: {self.method}"
    inputs = "\n".join(_indent(str(value)) for value in self.inputs)
    outputs = "\n".join(_indent(str(value)) for value in self.outputs)
    tolerances = _indent(f"rtol={self.rtol}, atol={self.atol}")
    body = f"Inputs:\n{inputs}\nOutputs:\n{outputs}\nTolerances:\n{tolerances}"
    result = f"{header}\n{_indent(body)}"

    np.set_printoptions(**prior_printoptions)
    return result


class Trace:
  """Stores the inputs and outputs of a series of calls to a module."""

  def __init__(self, module, function):
    """Extracts metadata from module and function and initializes.

    Example usage:
      def forward_pass(...):
        ...
      module = IreeCompiledModule(...)
      trace = Trace(module, forward_pass)
      forward_pass(TracedModule(module, trace))

    Args:
      module: the module who's outputs this trace will record.
      function: the function that module will be traced on.
    """
    # Extract metadata from module and function.
    self.module_name = module.module_name
    self.backend = module.backend
    self.function_name = function.__name__
    self.function_sourcefile = inspect.getsourcefile(function)
    source, start_line = inspect.getsourcelines(function)
    self.function_line_numbers = (start_line, start_line + len(source))
    self.function_source = "".join(source)

    self.calls = []

  def __str__(self):
    header = (f"Trace of {self.module_name} compiled to '{self.backend}' "
              f"on function '{self.function_name}':")
    # Give each call a number so it's easier to compare between multiple traces.
    calls = [f"{i + 1}. {str(call)}" for i, call in enumerate(self.calls)]
    calls = _indent("\n".join(calls))
    return f"{header}\n{calls}"

  def __iter__(self):
    for call in self.calls:
      yield call

  @staticmethod
  def compare_traces(ref_trace, tar_trace):
    traces_match = True

    # Check that all method invocations match.
    ref_methods = [(call.method, call.rtol, call.atol) for call in ref_trace]
    tar_methods = [(call.method, call.rtol, call.atol) for call in tar_trace]
    if ref_methods != tar_methods:
      # Raise a ValueError instead of returning False since this is an
      # unexpected error.
      raise ValueError(
          "The reference and target traces have different call structures:\n"
          f"Reference: {ref_methods}\nTarget:    {tar_methods}")

    for ref_call, tar_call in zip(ref_trace, tar_trace):
      logging.info("Comparing calls to '%s'", ref_call.method)
      rtol, atol = ref_call.get_tolerances()

      inputs_match = Trace._check_same(ref_call.inputs, tar_call.inputs, rtol,
                                       atol)
      if not inputs_match:
        logging.error("Inputs did not match.")
      outputs_match = Trace._check_same(ref_call.outputs, tar_call.outputs,
                                        rtol, atol)
      if not outputs_match:
        logging.error("Outputs did not match.")
      calls_match = inputs_match and outputs_match

      if not calls_match:
        logging.error("Comparision between '%s' and '%s' failed on method '%s'",
                      ref_trace.backend, tar_trace.backend, ref_call.method)
        logging.error("Reference call '%s':\n%s", ref_trace.backend, ref_call)
        logging.error("Target call '%s':\n%s", tar_trace.backend, tar_call)

      traces_match = traces_match and calls_match
    return traces_match

  @staticmethod
  def _check_same(ref, tar, rtol, atol):
    """Checks that ref and tar have identical datastructures and values."""
    # Check for matching types.
    if not isinstance(tar, type(ref)):
      logging.error(
          "Expected ref and tar to have the same type but got '%s' and '%s'",
          type(ref), type(tar))
      return False

    if ref is None:
      # Nothing to compare (e.g. the called method had no outputs).
      return True

    # Recursive check for dicts.
    if isinstance(ref, dict):
      if ref.keys() != tar.keys():
        logging.error(
            "Expected ref and tar to have the same keys, but got '%s' and '%s'",
            ref.keys(), tar.keys())
        return False
      # Check that all of the dictionaries' values are the same.
      for key in ref:
        if not Trace._check_same(ref[key], tar[key], rtol, atol):
          return False

    # Recursive check for iterables.
    elif isinstance(ref, list) or isinstance(ref, tuple):
      if len(ref) != len(tar):
        logging.error(
            "Expected ref and tar to have the same length, but got %s and %s",
            len(ref), len(tar))
        return False
      # Check that all of the iterables' values are the same.
      for i in range(len(ref)):
        if not Trace._check_same(ref[i], tar[i], rtol, atol):
          return False

    # Base check for numpy arrays.
    elif isinstance(ref, np.ndarray):
      if ref.dtype != tar.dtype:
        logging.error("Expected ref and tar to have the same dtype, but got %s "
                      " and %s", ref.dtype, tar.dtype)
        return False
      if np.issubdtype(ref.dtype, np.floating):
        return np.allclose(ref, tar, rtol=rtol, atol=atol)
      else:
        return np.array_equal(ref, tar)

    # Base check for native number types.
    elif isinstance(ref, (int, float)):
      return ref == tar

    # If outputs end up here then an extra branch for that type should be added.
    else:
      raise TypeError(f"Encountered results with unexpected type {type(ref)}")
    return True

  def _get_trace_dir(self, artifacts_dir):
    trace_dir = os.path.join(artifacts_dir, "traces")
    if not os.path.exists(trace_dir):
      os.makedirs(trace_dir)
    return trace_dir

  def save_plaintext(self, artifacts_dir, summarize=True):
    """Saves a human-readable string representation of this trace to disk.

    Args:
      artifacts_dir: the base directory to save the trace in.
      summarize: a bool controlling whether numpy should summarize the inputs
        and outputs if they're large. Setting this to False is very slow for
        large outputs.
    """
    prior_printoptions = np.get_printoptions()
    np.set_printoptions(
        linewidth=NUMPY_LINEWIDTH,
        threshold=None if summarize else sys.maxsize,
        edgeitems=10)  # Can show more items since they won't clutter the logs.

    trace_dir = self._get_trace_dir(artifacts_dir)
    path = os.path.join(trace_dir, f"{self.function_name}__{self.backend}.txt")
    with open(path, "w") as f:
      f.write(str(self))
      f.write("\n")

    np.set_printoptions(**prior_printoptions)


class TracedModule:

  def __init__(self, module, trace):
    """Wraps a CompiledModule so that all inputs and outputs are traced.

    The TracedModule returned will have an API almost identical to that of the
    passed CompiledModule. The only changes is that if the keywords `rtol` or
    `atol` are passed to one of the CompiledModule's methods, then they will be
    used to set the tolerance for comparing that call to the same call in
    another trace. So for example, calling `traced_module.add(a, b rtol=1e-8)`
    would be the same as calling `module.add(a, b)`.

    Args:
      module: the CompiledModule to trace.
      trace: the Trace to record calls to this module with.
    """
    self._module = module
    self._trace = trace

  def _trace_call(self, method, method_name):
    """Decorates a CompiledModule method to capture its inputs and outputs."""

    def call(*args, **kwargs):
      # Pop manually specified tolerances from the kwargs (if any).
      tolerances = {}
      tolerances["rtol"] = kwargs.pop("rtol", None)
      tolerances["atol"] = kwargs.pop("atol", None)
      # Only pass these to ModuleCall if they were specified by the user.
      tolerances = {k: v for k, v in tolerances.items() if v is not None}

      # Run the method and record the details of the call.
      outputs = method(*args, **kwargs)
      self._trace.calls.append(
          ModuleCall(method_name, args, outputs, **tolerances))
      return outputs

    return call

  def __getattr__(self, attr):
    # Try to resolve it as an attr on self._module.
    if not hasattr(self._module, attr):
      raise AttributeError(f"The compiled module does not have attr '{attr}'")
    module_attr = getattr(self._module, attr)
    if not hasattr(module_attr, "__call__"):
      # e.g. traced_module.backend
      return module_attr
    else:
      # e.g. traced_module.simple_mul(a, b)
      return self._trace_call(module_attr, method_name=attr)


class TracedModuleTestCase(tf.test.TestCase):
  """Compiles a tf.Module to multiple backends to test their correctness."""
  # Will be initialized by the @compile_module decorator.
  _module_class = None
  _exported_names = ()

  # Will be initialized in setUpClass.
  _ref_module = None
  _tar_modules = None

  @classmethod
  def _compile(cls, backend_info):
    return backend_info.CompiledModule(cls._module_class, backend_info,
                                       cls._exported_names, cls._artifacts_dir)

  @classmethod
  def setUpClass(cls):
    # Ran before any of the unit tests.
    super().setUpClass()
    if cls._module_class is None:
      raise AttributeError(
          "setUpClass was called but no module was specified. Specify a module "
          "to compile via the @tf_test_utils.compile_module decorator.")

    # Setup the directory for saving compilation artifacts and traces.
    cls._artifacts_dir = _setup_artifacts_dir(cls._module_class.__name__)

    # Setup crash reproducer for the test.
    crash_reproducer_path = os.path.join(cls._artifacts_dir, "reproducer.mlir")
    compiler.Context.default_crash_reproducer_path = crash_reproducer_path

    # Create a CompiledModule for the reference backend and each target backend.
    try:
      ref_backend_info = tf_utils.BackendInfo.ALL[FLAGS.reference_backend]
      cls._ref_module = cls._compile(ref_backend_info)

      tar_backend_infos = get_target_backends()
      cls._tar_modules = [
          cls._compile(backend_info) for backend_info in tar_backend_infos
      ]
    finally:
      # TODO(meadowlark): Move this into tf_util.compile_tf_module to prevent
      # overwritting `reproducer.mlir`.
      # Disable crash reproducer (to avoid inadvertently overwriting this
      # path if there are multiple TestCases in the same file).
      compiler.Context.default_crash_reproducer_path = None

  def setUp(self):
    # Ran before each unit test.
    super().setUp()
    self._ref_module.create_reinitialized()
    self._tar_modules = [
        module.create_reinitialized() for module in self._tar_modules
    ]

  def compare_backends(self, trace_function):
    """Run the reference and target backends on trace_function and compare them.

    Args:
      trace_function: a function accepting a TracedModule as its argument.
    """
    # Create Traces for each backend.
    ref_trace = Trace(self._ref_module, trace_function)
    tar_traces = [Trace(module, trace_function) for module in self._tar_modules]

    # Run the traces through trace_function with their associated modules.
    trace_function(TracedModule(self._ref_module, ref_trace))
    for module, trace in zip(self._tar_modules, tar_traces):
      trace_function(TracedModule(module, trace))

    # Compare each target trace of trace_function with the reference trace.
    failed_backend_indices = []
    for i, tar_trace in enumerate(tar_traces):
      logging.info("Comparing the reference backend '%s' with '%s'",
                   ref_trace.backend, tar_trace.backend)
      traces_match = Trace.compare_traces(ref_trace, tar_trace)
      if not traces_match:
        failed_backend_indices.append(i)

    # Save the results to disk before validating.
    ref_trace.save_plaintext(self._artifacts_dir, FLAGS.summarize)
    for tar_trace in tar_traces:
      tar_trace.save_plaintext(self._artifacts_dir, FLAGS.summarize)

    # Validate results.
    if len(failed_backend_indices) > 0:
      # Extract info for logging.
      failed_backends = [tar_traces[i].backend for i in failed_backend_indices]
      failure_info = (
          "Comparision between the reference backend and the following targets "
          f"failed: {failed_backends}. The errors above show the inputs and "
          "outputs the non-matching calls.")

      # This condition is always True, but is useful for context in the logs.
      self.assertFalse(len(failed_backends) > 0, failure_info)

  @classmethod
  def tearDownClass(cls):
    # Ran after all unit tests are completed.
    super().tearDownClass()
