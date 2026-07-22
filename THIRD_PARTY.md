# Third-Party Artifact Dependencies

SCARF itself is distributed under the repository `LICENSE`. The following tools,
models, datasets, and submodules remain under their respective licenses. They
are downloaded or referenced by the artifact and are not relicensed as SCARF.

| Component | Pinned source | License / terms | Artifact use |
|---|---|---|---|
| TranSplat | gitlink `aaa29a40f96774b5debdc6b273d1e7512b14ca37` | MIT, LICENSE SHA256 `b6b3f85e0915abdd3d54017459e14d532ef9039e424e4febb55810b028e53f68` | Model implementation |
| MVSplat | gitlink `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01` | MIT, LICENSE SHA256 `b77d405e6481b8feff4c4769cdf06c380b23a27f1bde7b5607cbbd1bb632db51` | Model implementation |
| DepthSplat | gitlink `1f5e5486f005e5b9975cca5cbed3d61acf465707` | MIT, LICENSE SHA256 `03614e84680c6b412f52ada874a6b49b2fd1426878f909f036175573a3b4f49b` | Model implementation |
| Depth Anything V2 Base | Hugging Face revision `a4e71a6c2ce52fe50df0f212066b0d4a87be9b5e`, weights SHA256 `0d2b7002e62d39d655571c371333340bd88f67ab95050c03591555aa05645328` | CC-BY-NC-4.0 | TranSplat depth-prior backbone; downloaded separately and excluded from release archives |
| DINOv2 | commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | Apache-2.0, LICENSE SHA256 `600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7` | Pinned local DepthSplat backbone source; full backbone weights come from the pinned DepthSplat checkpoints |
| iFlow | `https://gitee.com/oscc-project/iFlow`, commit `04b4d98` | Mulan PSL v2 | RTL-to-GDS orchestration |
| ASAP7 | `https://asap.asu.edu/asap/` via pinned iFlow collateral | ASAP7 distribution terms | Predictive 7 nm physical proxy |
| DeepScaleTool | `hardware/scaling/vendor/DeepScaleTool.xlsm` | GPLv3, workbook SHA256 `561a3f8f5e91a3c496d6e0f4262c09412209f3bbc6d22b714cde323ebd958df8` | Technology normalization |
| Ramulator 2 | tag `v2.1.0`, commit `38c51d40a976c6b07fbc09de869a7e08dc187d29` | BSD-3-Clause, LICENSE SHA256 `74a1f19701bb06cb80b4ec31e6de6472f5fd767e5949c1c4d51f70007d18b3f2` | Public LPDDR5 timing proxy |
| DRAMPower | tag `v6.0.2`, commit `c26ccf948100199977533ee8198ce8ec7fbd83ac` | BSD-3-Clause, LICENSE SHA256 `70ece834787b0c70141d2e5a6d0f2e09dd998dd3b113af6b2aa296e58a3c6024` | Public LPDDR5 energy proxy |
| Re10K | Official project and prepared evaluation subset | No redistribution license is claimed | Evaluation data downloaded separately |
| ACID | Official project and prepared evaluation subset | No redistribution license is claimed | Evaluation data downloaded separately |
| DL3DV | Gated benchmark revision `9684e8382278c5e18173c1e72bd246daf2874539` | Gated access does not grant redistribution rights | Separate native 270 by 480 and Re10K-compatible 360 by 640 trees |

Dataset download sources, terms pages, and tree-hash requirements are
machine-readable in `artifact/manifests/datasets.json`. The release checker
rejects the archive until every prepared tree is verified. DL3DV has separate
native and Re10K-compatible tree hashes. No entry grants permission to
redistribute dataset content. Re10K and ACID use the public evaluation-only
preprocessing mirror linked by the MVSplat and pixelSplat documentation.

## DeepScaleTool Reference

The pinned workbook downloaded on 2026-07-13 has SHA256:

```text
561a3f8f5e91a3c496d6e0f4262c09412209f3bbc6d22b714cde323ebd958df8
```

The non-interactive implementation in the artifact is checked against the workbook
tables and the examples in Sarangi and Baas, *DeepScaleTool: A Tool for the
Accurate Estimation of Technology Scaling in the Deep-Submicron Era*, ISCAS
2021, DOI `10.1109/ISCAS51556.2021.9401196`.

## Release Requirements

Before archival release, `scripts/check_release.py` must verify all gitlinks,
URLs, expected hashes, and required license notices. Third-party checkpoints and
dataset samples are not included in the Zenodo package. The archive contains
download instructions and hashes.
