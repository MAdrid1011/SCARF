FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        g++ \
        gcc \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

FROM base AS source-manifest

WORKDIR /work/scarf
COPY . /work/scarf

# Build from the reviewed source bundle. Published source archives carry their
# manifest already; Git worktrees create the same bundle during the build.
RUN if [ -f release-manifest.json ]; then \
        mkdir -p /opt/release/SCARF-AE \
        && cp -a /work/scarf/. /opt/release/SCARF-AE/; \
    else \
        python3 scripts/build_archive.py --source-only --require-doi \
            --output /tmp/scarf-source.tar.gz --prefix SCARF-AE \
        && mkdir -p /opt/release \
        && tar -xzf /tmp/scarf-source.tar.gz -C /opt/release \
        && rm -f /tmp/scarf-source.tar.gz; \
    fi \
    && test -f /opt/release/SCARF-AE/release-manifest.json

FROM base AS runtime

WORKDIR /opt/scarf
COPY --from=source-manifest /opt/release/SCARF-AE/ /opt/scarf/

RUN bash install.sh --profile classic --venv /opt/scarf/.venv/classic \
    && mkdir -p /results /opt/scarf/mvsplat/checkpoints \
    && chmod -R a+rwX /opt/scarf /results

ENV SCARF_PYTHON_CLASSIC=/opt/scarf/.venv/classic/bin/python \
    SCARF_OUTPUT_ROOT=/results \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

ENTRYPOINT ["/opt/scarf/docker/run-functional.sh"]
