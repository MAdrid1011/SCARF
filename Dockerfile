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

WORKDIR /opt/scarf
COPY . /opt/scarf

RUN python3 scripts/build_archive.py --source-only --require-doi \
        --output /tmp/scarf-source.tar.gz --prefix SCARF-AE \
    && mkdir /tmp/scarf-source \
    && tar -xzf /tmp/scarf-source.tar.gz -C /tmp/scarf-source \
    && cp /tmp/scarf-source/SCARF-AE/release-manifest.json /opt/scarf/ \
    && rm -rf /tmp/scarf-source /tmp/scarf-source.tar.gz \
    && find /opt/scarf -type d -name .git -prune -exec rm -rf {} + \
    && find /opt/scarf -type f -path '*/.git' -delete

FROM base AS runtime

WORKDIR /opt/scarf
COPY --from=source-manifest /opt/scarf /opt/scarf

RUN bash install.sh --profile classic --venv /opt/scarf/.venv/classic \
    && mkdir -p /results /opt/scarf/mvsplat/checkpoints \
    && chmod -R a+rwX /opt/scarf /results

ENV SCARF_PYTHON_CLASSIC=/opt/scarf/.venv/classic/bin/python \
    SCARF_OUTPUT_ROOT=/results

ENTRYPOINT ["/opt/scarf/docker/run-functional.sh"]
