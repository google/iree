#!/usr/bin/env python3
# Copyright 2021 Google LLC
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
"""Posts benchmark results to GitHub as pull request comments.

This script is meant to be used by Buildkite for automation. It requires the
following environment to be set:

- BUILDKITE_BUILD_URL: the link to the current Buildkite build.
- BUILDKITE_COMMIT: the pull request HEAD commit.
- BUILDKITE_PULL_REQUEST: the current pull request number.
- GITHUB_TOKEN: personal access token to authenticate against GitHub API.

This script uses pip package "markdown_strings".

Example usage:
  # Export necessary environment variables:
  export ...
  # Then run the script:
  python3 post_benchmarks_as_pr_comment.py <benchmark-json-file>...
  #   where each <benchmark-json-file> is expected to be of format:
  #   {"commit": <commit-sha>, "benchmarks": {<name>: <value>, ...}}
"""

import argparse
import json
import os
import requests
import markdown_strings as md

GITHUB_IREE_API_PREFIX = "https://api.github.com/repos/google/iree"


def get_required_env_var(var):
  """Gets the value for a required environment variable."""
  value = os.getenv(var, None)
  if value is None:
    raise RuntimeError(f'Missing environment variable "{var}"')
  return value


def aggregate_all_benchmarks(benchmark_files):
  """Aggregates all benchmarks in the given files.

  Args:
  - benchmark_files: A list of JSON files, each with the format:
    {"commit": <commit-sha>, "benchmarks": {<name>: <value>, ...}}

  Returns:
  - A list of benchmark (name, value) tuples.
  """
  pr_commit = get_required_env_var("BUILDKITE_COMMIT")
  all_benchmarks = []

  for benchmark_file in benchmark_files:
    with open(benchmark_file) as f:
      benchmarks = json.load(f)
    if benchmarks["commit"] != pr_commit:
      raise ValueError("Inconsistent pull request commit")
    for name, value in benchmarks["benchmarks"].items():
      if name in all_benchmarks:
        raise ValueError(f"Duplicated benchmarks: {name}")
      all_benchmarks.append((name, value))

  return sorted(all_benchmarks)


def get_benchmark_result_markdown(benchmark_files):
  """Gets markdown summary of all benchmarks in the given files."""
  all_benchmarks = aggregate_all_benchmarks(benchmark_files)
  names = [k for k, _ in all_benchmarks]
  values = [v for _, v in all_benchmarks]
  names.insert(0, "Benchmark Name")
  values.insert(0, "Time (ms)")

  build_url = get_required_env_var("BUILDKITE_BUILD_URL")
  pr_commit = get_required_env_var("BUILDKITE_COMMIT")

  commit = f"@ commit {pr_commit}"
  header = md.header("Benchmark results", 3)
  benchmark_table = md.table([names, values])
  link = "See more details on " + md.link("Buildkite", build_url)

  return "\n\n".join([header, commit, benchmark_table, link])


def comment_on_pr(content):
  """Posts the given content as comments to the current pull request."""
  pr_number = get_required_env_var("BUILDKITE_PULL_REQUEST")
  # Buildkite sets this to "false" if not running on a PR:
  # https://buildkite.com/docs/pipelines/environment-variables#bk-env-vars-buildkite-pull-request
  if pr_number == "false":
    raise ValueError("Not a pull request")

  api_token = get_required_env_var('GITHUB_TOKEN')
  headers = {
      "Accept": "application/vnd.github.v3+json",
      "Authorization": f"token {api_token}",
  }
  payload = json.dumps({"event": "COMMENT", "body": content})

  api_endpoint = f"{GITHUB_IREE_API_PREFIX}/pulls/{pr_number}/reviews"
  request = requests.post(api_endpoint, data=payload, headers=headers)
  if request.status_code != 200:
    raise requests.RequestException(
        f"Failed to comment on GitHub; error code: {request.status_code}")


def parse_arguments():
  """Parses command-line options."""

  def check_file_path(path):
    if os.path.isfile(path):
      return path
    else:
      raise ValueError(path)

  parser = argparse.ArgumentParser()
  parser.add_argument("benchmark_files",
                      metavar="<benchmark-json-file>",
                      type=check_file_path,
                      nargs="+",
                      help="Path to the JSON file containing benchmark results")
  parser.add_argument("--dry-run",
                      action="store_true",
                      help="Print the comment instead of posting to GitHub")
  args = parser.parse_args()

  return args


def main(args):
  benchmarks_md = get_benchmark_result_markdown(args.benchmark_files)

  if args.dry_run:
    print(benchmarks_md)
  else:
    comment_on_pr(benchmarks_md)


if __name__ == "__main__":
  main(parse_arguments())
