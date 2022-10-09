# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Script to generate a HTML summary of SHARK Tank benchmarks.

Example usage:

python parse_shark_benchmarks.py \
  --shark_version=20220924.162 \
  --iree_version=20220924.276 \
  --cpu_shark_csv=icelake_shark_bench_results.csv \
  --cpu_iree_csv=icelake_iree_bench_results.csv \
  --gpu_shark_csv=a100_shark_bench_results.csv \
  --gpu_iree_csv=a100_iree_bench_results.csv \
  --output_path=/tmp/summary.html

"""

import argparse
import pandas as pd
import pathlib
import sys

from datetime import datetime

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent / ".." / ".." / "python"))
from reporting.common import html_utils

# Column headers.
_MODEL = "model"
_DIALECT = "dialect"
_DATA_TYPE = "data_type"
_BASELINE = "baseline"
_DEVICE = "device"
_BASELINE_LATENCY = "baseline latency (ms)"
_IREE_LATENCY = "IREE latency (ms)"
_SHARK_LATENCY = "SHARK latency (ms)"
_IREE_VS_BASELINE = "IREE vs baseline"
_SHARK_VS_BASELINE = "SHARK vs baseline"
_IREE_VS_SHARK = "IREE vs SHARK"

_PERF_COLUMNS = [_IREE_VS_BASELINE, _SHARK_VS_BASELINE, _IREE_VS_SHARK]
_LATENCY_COLUMNS = [_BASELINE_LATENCY, _IREE_LATENCY, _SHARK_LATENCY]


def _generate_table(df_iree, df_shark, df_baseline, title):
  """Generates a table comparing latencies between IREE, SHARK and a baseline."""
  summary = pd.DataFrame(columns=[
      _MODEL, _BASELINE, _DATA_TYPE, _DIALECT, _DEVICE, _BASELINE_LATENCY,
      _IREE_LATENCY, _SHARK_LATENCY, _IREE_VS_BASELINE, _SHARK_VS_BASELINE,
      _IREE_VS_SHARK
  ])

  models = df_iree.model.unique()
  for model in models:
    iree_results_per_model = df_iree.loc[df_iree.model == model]
    dialects = iree_results_per_model.dialect.unique()
    for dialect in dialects:
      iree_results_per_dialect = iree_results_per_model.loc[
          iree_results_per_model.dialect == dialect]
      data_types = iree_results_per_dialect.data_type.unique()
      for data_type in data_types:
        iree_results_per_datatype = iree_results_per_dialect.loc[
            iree_results_per_dialect.data_type == data_type]
        device_types = iree_results_per_datatype.device.unique()
        for device in device_types:
          iree_results = iree_results_per_datatype.loc[
              iree_results_per_datatype.device == device]
          if len(iree_results) != 3:
            print("Warning! Expected number of results to be 3. Got " +
                  str(len(iree_results)))
            print(str(iree_results))
            continue

          baseline_results = df_baseline.loc[(df_baseline.model == model) &
                                             (df_baseline.dialect == dialect) &
                                             (df_baseline.data_type
                                              == data_type) &
                                             (df_baseline.device == device)]

          if baseline_results.empty:
            # We use snapshots of latencies for baseline. If it is a new
            # benchmark that is not included in the snapshot yet, emit a
            # warning.
            print(
                "Warning: No baseline results found for " + model + ", " +
                dialect + ", " + data_type + ", " + device +
                ". Using IREE version as baseline. Please update baseline csv.")
            engine = iree_results.engine.iloc[0]
            baseline_latency = iree_results.loc[iree_results.engine == engine]
            baseline_latency = baseline_latency.iloc[0]["ms/iter"]
          else:
            engine = baseline_results.engine.iloc[0]
            baseline_latency = baseline_results.loc[baseline_results.engine ==
                                                    engine]
            baseline_latency = baseline_latency.iloc[0]["ms/iter"]

          iree_latency = iree_results.loc[iree_results.engine == "shark_iree_c"]
          iree_latency = iree_latency.iloc[0]["ms/iter"]
          iree_vs_baseline = html_utils.format_latency_comparison(
              iree_latency, baseline_latency)

          if df_shark is not None:
            shark_results = df_shark.loc[(df_shark.model == model) &
                                         (df_shark.dialect == dialect) &
                                         (df_shark.data_type == data_type) &
                                         (df_shark.device == device)]

            shark_latency = shark_results.loc[shark_results.engine ==
                                              "shark_iree_c"]
            shark_latency = shark_latency.iloc[0]["ms/iter"]
            shark_vs_baseline = html_utils.format_latency_comparison(
                shark_latency, baseline_latency)
            iree_vs_shark = html_utils.format_latency_comparison(
                iree_latency, shark_latency)
          else:
            # If there are no SHARK benchmarks available, use default values.
            # These columns will be hidden later.
            shark_latency = 0
            shark_vs_baseline = "1.0x faster"
            iree_vs_shark = "1.0x faster"

          summary.loc[len(summary)] = [
              model, engine, data_type, dialect, device,
              "{:.1f}".format(baseline_latency), "{:.1f}".format(iree_latency),
              "{:.1f}".format(shark_latency), iree_vs_baseline,
              shark_vs_baseline, iree_vs_shark
          ]

  summary = summary.round(2)

  st = summary.style.set_table_styles(html_utils.get_table_css())
  st = st.hide(axis="index")
  if df_shark is None:
    st = st.hide_columns(
        subset=[_SHARK_LATENCY, _SHARK_VS_BASELINE, _IREE_VS_SHARK])
  st = st.set_caption(title)
  st = st.applymap(html_utils.style_performance, subset=_PERF_COLUMNS)
  st = st.set_properties(subset=[_MODEL],
                         **{
                             "width": "300px",
                             "text-align": "left",
                         })
  st = st.set_properties(subset=[_BASELINE],
                         **{
                             "width": "140",
                             "text-align": "center",
                         })
  st = st.set_properties(subset=[_DIALECT, _DATA_TYPE, _DEVICE],
                         **{
                             "width": "100",
                             "text-align": "center",
                         })
  st = st.set_properties(subset=_LATENCY_COLUMNS,
                         **{
                             "width": "100",
                             "text-align": "right",
                         })
  st = st.set_properties(subset=_PERF_COLUMNS,
                         **{
                             "width": "150px",
                             "text-align": "right",
                             "color": "#ffffff"
                         })

  return st.to_html() + "<br/>"


def generate_table(iree_csv,
                   baseline_csv,
                   shark_csv=None,
                   shape_type="static",
                   device="cpu",
                   title="Benchmarks"):
  """Generates a table comparing latencies between IREE, SHARK and a baseline.

  Args:
    iree_csv: Path to the csv file containing IREE latencies.
    baseline_csv: Path to the csv file containing baseline latencies.
    shark_csv: Path to the csv file containing SHARK-Runtime latencies. This is optional.
    shape_type: Currently either `static` or `dynamic`.
    device: Device used to run the benchmarks.
    title: The title of the generated table.

  Returns:
    A HTML file containing the summarized report.
  """
  shark_df = None
  if shark_csv is not None:
    shark_df = pd.read_csv(shark_csv)
    shark_df = shark_df.loc[(shark_df.shape_type == shape_type) &
                            (shark_df.device == device)]

  iree_df = pd.read_csv(iree_csv)
  iree_df = iree_df.loc[(iree_df.shape_type == shape_type) &
                        (iree_df.device == device)]

  baseline_df = pd.read_csv(baseline_csv)
  baseline_df = baseline_df.loc[(baseline_df.shape_type == shape_type) &
                                (baseline_df.device == device)]

  return _generate_table(iree_df, shark_df, baseline_df, title)


def main(args):
  """Summarizes benchmark results generated by the SHARK Tank."""
  verison_html = "<i>IREE version: " + args.iree_version + "</i><br/>"
  verison_html = verison_html + "<i>SHARK version: " + args.shark_version + "</i><br/>"
  verison_html = verison_html + "<i>last updated: " + datetime.today().strftime(
      "%Y-%m-%d") + "</i><br/><br/>"
  html = html_utils.generate_header_and_legend(verison_html)

  # Generate Server CPU Static.
  html = html + generate_table(
      args.cpu_iree_csv,
      args.cpu_baseline_csv,
      shark_csv=args.cpu_shark_csv,
      shape_type="static",
      device="cpu",
      title="Server Intel Ice Lake CPU (Static Shapes)")

  # Generate Server GPU Static.
  html = html + generate_table(
      args.gpu_iree_csv,
      args.gpu_baseline_csv,
      shark_csv=args.gpu_shark_csv,
      shape_type="static",
      device="cuda",
      title="Server NVIDIA Tesla A100 GPU (Static Shapes)")

  # Generate Server CPU Dynamic.
  html = html + generate_table(
      args.cpu_iree_csv,
      args.cpu_baseline_csv,
      shark_csv=args.cpu_shark_csv,
      shape_type="dynamic",
      device="cpu",
      title="Server Intel Ice Lake CPU (Dynamic Shapes)")

  # Generate Server GPU Dynamic.
  html = html + generate_table(
      args.gpu_iree_csv,
      args.gpu_baseline_csv,
      shark_csv=args.gpu_shark_csv,
      shape_type="dynamic",
      device="cuda",
      title="Server NVIDIA Tesla A100 GPU (Dynamic Shapes)")

  args.output_path.write_text(html)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--cpu_shark_csv",
      type=str,
      default=None,
      help=
      "The path to the csv file with CPU benchmarking results from the SHARK runtime."
  )
  parser.add_argument(
      "--cpu_iree_csv",
      type=str,
      default=None,
      help="The path to the csv file with CPU benchmarking results from IREE.")
  parser.add_argument(
      "--cpu_baseline_csv",
      type=str,
      default="data/icelake_baseline_0919.csv",
      help="The path to the csv file containing baseline CPU results.")
  parser.add_argument(
      "--gpu_shark_csv",
      type=str,
      default=None,
      help=
      "The path to the csv file with GPU benchmarking results from the SHARK runtime."
  )
  parser.add_argument(
      "--gpu_iree_csv",
      type=str,
      default=None,
      help="The path to the csv file with CPU benchmarking results from IREE.")
  parser.add_argument(
      "--gpu_baseline_csv",
      type=str,
      default="data/a100_baseline_0919.csv",
      help="The path to the csv file containing baseline GPU results.")
  parser.add_argument("--iree_version",
                      type=str,
                      default=None,
                      help="The IREE version.")
  parser.add_argument("--shark_version",
                      type=str,
                      default=None,
                      help="The SHARK version.")
  parser.add_argument(
      "--output_path",
      type=pathlib.Path,
      default="/tmp/summary.html",
      help="The path to the output html file that summarizes results.")
  return parser.parse_args()


if __name__ == "__main__":
  main(parse_args())
