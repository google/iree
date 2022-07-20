# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

# Sets up files and environment to enable running scripts in docker.
# In particular, does some shenanigans to enable running with the current user.
# Some of this setup is only strictly necessary for Bazel, but it doesn't hurt
# for anything else.
# Requires that DOCKER_WORKDIR and DOCKER_TMPDIR have been set
function docker_run() {
    # Make the source repository available and launch containers in that
    # directory.
    DOCKER_RUN_ARGS=(
      --volume="${DOCKER_WORKDIR}:${DOCKER_WORKDIR}"
      --workdir="${DOCKER_WORKDIR}"
    )

    # Delete the container after the run is complete.
    DOCKER_RUN_ARGS+=(--rm)


    # Run as the current user and group. If only it were this simple...
    DOCKER_RUN_ARGS+=(--user="$(id -u):$(id -g)")


    # The Docker container doesn't know about the users and groups of the host
    # system. We have to tell it. This is just a mapping of IDs to names though.
    # The thing that really matters is the IDs, so the key thing is that Docker
    # writes files as the same ID as the current user, which we set above, but
    # without the group and passwd file, lots of things get upset because they
    # don't recognize the current user ID (e.g. `whoami` fails). Bazel in
    # particular looks for a home directory and is not happy when it can't find
    # one.
    # So we make the container share the host mapping, which guarantees that the
    # current user is mapped. If there was any user or group in the container
    # that we cared about, this wouldn't necessarily work because the host and
    # container don't necessarily map the ID to the same user. Luckily though,
    # we don't.
    # We don't just mount the real /etc/passwd and /etc/group because Google
    # Linux workstations do some interesting stuff with user/group permissions
    # such that they don't contain the information about normal users and we
    # want these scripts to be runnable locally for debugging.
    # Instead we dump the results of `getent` to some fake files.
    local fake_etc_dir="${DOCKER_TMPDIR}/fake_etc"
    mkdir -p "${fake_etc_dir?}"

    local fake_group="${fake_etc_dir?}/group"
    local fake_passwd="${fake_etc_dir?}/passwd"

    getent group > "${fake_group?}"
    getent passwd > "${fake_passwd?}"

    DOCKER_RUN_ARGS+=(
      --volume="${fake_group?}:/etc/group:ro"
      --volume="${fake_passwd?}:/etc/passwd:ro"
    )


    # Bazel stores its cache in the user home directory by default. It's
    # possible to override this, but that would require changing our Bazel
    # startup options, which means polluting all our scripts and making them not
    # runnable locally. Instead, we give it a special home directory to write
    # into. We don't just mount the user home directory (or some subset thereof)
    # for two reasons:
    #   1. We probably don't want Docker to just write into the user's home
    #      directory when running locally.
    #   2. This allows us to control the device the home directory is in. Bazel
    #      tends to be IO bound at even moderate levels of CPU parallelism and
    #      the difference between a persistent SSD and a local scratch SSD can
    #      be huge. In particular, Kokoro has the home directory on the former
    #      and the work directory on the latter.
    local fake_home_dir="${DOCKER_TMPDIR}/fake_home"
    mkdir -p "${fake_home_dir}"

    DOCKER_RUN_ARGS+=(
      --volume="${fake_home_dir?}:${HOME?}"
    )

    # Make gcloud credentials available. This isn't necessary when running in
    # GCE but enables using this script locally with remote caching.
    DOCKER_RUN_ARGS+=(
      --volume="${HOME?}/.config/gcloud:${HOME?}/.config/gcloud:ro"
    )

    docker run "${DOCKER_RUN_ARGS[@]}" "$@"
}

docker_run "$@"
