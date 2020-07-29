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

import collections
import os
import re
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
  if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)
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


class TracedModuleTestCase(tf.test.TestCase):
  """Compiles a tf.Module to multiple backends to test their correctness."""
  # This class uses the following abbreviations:
  #   ref: reference – for the reference CompiledModule
  #   tar: target - for one of the target CompiledModules

  # Will be initialized by the @compile_module decorator.
  _module_class = None
  _exported_names = ()

  # Will be initialized in setUpClass
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
    # Trace the test function for each backend.
    ref_trace = tf_utils.TracedModule(self._ref_module, trace_function)
    tar_traces = [
        tf_utils.TracedModule(module, trace_function)
        for module in self._tar_modules
    ]

    # Compare each target trace with the reference trace.
    failed_backend_indices = []
    for i, tar_trace in enumerate(tar_traces):
      logging.info("Comparing the reference backend '%s' with '%s'",
                   ref_trace.backend, tar_trace.backend)
      traces_match = self._compare_traces(ref_trace, tar_trace)
      if not traces_match:
        failed_backend_indices.append(i)

    # Save the results to disk before validating.
    trace_dir = os.path.join(self._artifacts_dir, "traces")
    if not os.path.exists(trace_dir):
      os.makedirs(trace_dir)
    ref_trace.save_plaintext(trace_dir, FLAGS.summarize)
    for tar_trace in tar_traces:
      tar_trace.save_plaintext(trace_dir, FLAGS.summarize)

    # Validate results.
    if len(failed_backend_indices) > 0:
      # Extract info for logging.
      failed_traces = [tar_traces[b] for b in failed_backend_indices]
      failed_backends = [trace.backend for trace in failed_traces]
      failure_info = (
          "Comparision between the reference backend and the following targets "
          f"failed: {failed_backends}. The  errors above show the outputs of "
          "the non-matching calls.")

      # This condition is always True, but is useful for context in the logs.
      self.assertFalse(len(failed_backends) > 0, failure_info)

  def _compare_traces(self, ref_trace, tar_trace):
    traces_match = True
    for ref_call, tar_call in zip(ref_trace, tar_trace):
      rtol, atol = ref_call.get_tolerances()

      logging.info("Comparing calls to '%s'", ref_call.method)
      calls_match = self._check_same(ref_call.outputs, tar_call.outputs, rtol,
                                     atol)

      # Log the inputs and outputs if they don't match.
      if not calls_match:
        logging.error("Comparision between '%s' and '%s' failed on method '%s'",
                      ref_trace.backend, tar_trace.backend, ref_call.method)
        logging.error("Reference result '%s':\n%s", ref_trace.backend, ref_call)
        logging.error("Target result '%s':\n%s", tar_trace.backend, tar_call)

      traces_match = traces_match and calls_match
    return traces_match

  def _check_same(self, ref, tar, rtol, atol):
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
        if not self._check_same(ref[key], tar[key], rtol, atol):
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
        if not self._check_same(ref[i], tar[i], rtol, atol):
          return False

    # Base check for numpy arrays.
    elif isinstance(ref, np.ndarray):
      if isinstance(ref.flat[0], np.floating):
        return np.allclose(ref, tar, rtol=rtol, atol=atol)
      else:
        return np.array_equal(ref, tar)

    # If outputs end up here then an extra branch for that type should be added.
    else:
      logging.warning("Comparing an unexpected result type of %s", type(ref))
      return ref == tar
    return True

  @classmethod
  def tearDownClass(cls):
    # Ran after all unit tests are completed.
    super().tearDownClass()
